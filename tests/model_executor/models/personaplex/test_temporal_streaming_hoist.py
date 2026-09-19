# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exactness regression tests for the RoPE-table / RingKV-index caching hoist.

These tests pin the *current* (pre-cache) behavior of ``_apply_rope`` and
``_RingKV.complete`` as a frozen oracle, verbatim-copied below, and compare it
against the real module every test run -- never against a pre-saved fixture
file. A saved-tensor golden file would be fragile across torch versions /
hardware (``cos``/``sin`` are not bit-portable), so both the "legacy" and the
"new" computation happen back to back, same process, same run, same random
draw.

The frozen copies below (``_legacy_apply_rope``, ``_LegacyRingKV``,
``_legacy_temporal_layer_forward``, ``_legacy_mimi_layer_forward``) are exact
copies of ``personaplex_temporal.py`` / ``personaplex_mimi.py`` as of the
pre-refactor state (git commit 64092757) and must NOT be updated when those
modules change -- they are the fixed reference this suite checks the refactor
against, not living code.

These tests target the *post-refactor* API described in the caching plan
(``_apply_rope(q, k, offset: int, ...)``, ``_RingKV.complete(k, v, offset:
int)``): they are expected to FAIL until that refactor lands, then pass
without any float-tolerance slop (``torch.equal`` throughout, since this is a
pure refactor and any mismatch means an ordering/accumulation bug).

Also (re-)run after the refactor:
    pytest tests/model_executor/models/personaplex/test_streaming_code2wav.py -v
    pytest tests/model_executor/models/personaplex/duplex/ -v
    pytest tests/model_executor/stage_input_processors/test_personaplex.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm_omni.model_executor.models.personaplex.personaplex_depformer import _rms_norm_f32
from vllm_omni.model_executor.models.personaplex.personaplex_mimi import (
    _MimiStreamingTransformer,
)
from vllm_omni.model_executor.models.personaplex.personaplex_temporal import (
    PersonaPlexTemporalStreaming,
    _apply_rope,
    _rope_tables,
    _ringkv_positions,
    _RingKV,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ===========================================================================
# Frozen oracle: verbatim pre-refactor copies (personaplex_temporal.py,
# commit 64092757). Do not "fix" these to match the new code -- they are the
# reference the new code is checked against.
# ===========================================================================


def _legacy_apply_rope(q: torch.Tensor, k: torch.Tensor, offset: torch.Tensor, max_period: float = 10_000.0):
    B, H, T, D = q.shape
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = offset.float() + torch.arange(T, device=q.device, dtype=torch.float32)
    ts = ts.view(1, -1, 1)

    dims = q.shape[:-1]
    q = q.view(*dims, D // 2, 2)
    k = k.view(*dims, D // 2, 2)
    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr
    dtype = q.dtype
    qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1)
    ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1)
    return qo.view(*dims, D), ko.view(*dims, D)


class _LegacyRingKV:
    def __init__(self, batch_size: int, num_heads: int, dim_per_head: int, capacity: int, device, dtype):
        self.capacity = capacity
        self.cache = torch.zeros((2, batch_size, num_heads, capacity, dim_per_head), device=device, dtype=dtype)
        self.end_offset = torch.zeros(1, device=device, dtype=torch.long)
        self.start_offset = torch.zeros(batch_size, device=device, dtype=torch.long)

    def reset(self) -> None:
        self.end_offset.zero_()
        self.start_offset.zero_()

    def reset_slot(self, b: int) -> None:
        self.start_offset[b] = self.end_offset.clone()

    def bump_slot_start(self, b: int) -> None:
        self.start_offset[b] += 1

    def complete(self, k: torch.Tensor, v: torch.Tensor):
        B, H, T, D = k.shape
        indexes = torch.arange(T, device=self.end_offset.device, dtype=self.end_offset.dtype) + self.end_offset
        indexes = indexes % self.capacity
        self.cache[0].index_copy_(2, indexes, k)
        self.cache[1].index_copy_(2, indexes, v)
        self.end_offset.add_(T)

        idx = torch.arange(self.capacity, device=self.end_offset.device, dtype=torch.long)
        invalid = idx >= self.end_offset
        end_index = self.end_offset % self.capacity
        delta = idx - end_index
        positions = torch.where(delta <= 0, self.end_offset + delta, self.end_offset + delta - self.capacity)
        positions = torch.where(invalid, torch.full_like(positions, -1), positions)
        positions = positions.view(1, -1)
        below = positions < self.start_offset.view(-1, 1)
        positions = torch.where(below, torch.full_like(positions, -1), positions)
        return self.cache[0], self.cache[1], positions


def _legacy_temporal_layer_forward(layer, x: torch.Tensor, kv: _LegacyRingKV, offset: torch.Tensor, context: int):
    """Frozen copy of ``_TemporalLayer.forward``, run against a REAL layer's
    own weights so legacy vs. new only ever differ by the cache refactor."""
    B, T, _ = x.shape
    h = _rms_norm_f32(x, layer.norm1_alpha, 1e-8)
    qkv = F.linear(h, layer.in_proj_weight)
    qkv = qkv.view(B, T, 3, layer.num_heads, layer.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = _legacy_apply_rope(q, k, offset)

    keys, values, pos_k = kv.complete(k, v)
    pos_k = pos_k.view(pos_k.shape[0], 1, pos_k.shape[1])
    pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(1, -1, 1)
    delta = pos_q - pos_k
    attn_bias = (pos_k >= 0) & (delta >= 0) & (delta < context)
    attn_bias = attn_bias.unsqueeze(1)
    attn = F.scaled_dot_product_attention(q, keys, values, attn_bias, dropout_p=0.0)
    attn = attn.transpose(1, 2).reshape(B, T, layer.dim)
    x = x + F.linear(attn, layer.out_proj_weight)

    h = _rms_norm_f32(x, layer.norm2_alpha, 1e-8)
    a, b = F.linear(h, layer.gating_in).chunk(2, dim=-1)
    return x + F.linear(F.silu(a) * b, layer.gating_out)


def _legacy_mimi_layer_forward(layer, x: torch.Tensor, kv: _LegacyRingKV, offset: torch.Tensor, context: int):
    """Frozen copy of ``_MimiTransformerLayer.forward`` (personaplex_mimi.py)."""
    B, T, _ = x.shape
    h = layer.norm1(x)
    qkv = F.linear(h, layer.in_proj_weight)
    qkv = qkv.view(B, T, 3, layer.num_heads, layer.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = _legacy_apply_rope(q, k, offset)
    keys, values, pos_k = kv.complete(k, v)
    pos_k = pos_k.view(pos_k.shape[0], 1, pos_k.shape[1])
    pos_q = offset + torch.arange(T, device=q.device, dtype=torch.long).view(1, -1, 1)
    delta = pos_q - pos_k
    attn_bias = (pos_k >= 0) & (delta >= 0) & (delta < context)
    attn = F.scaled_dot_product_attention(q, keys, values, attn_bias.unsqueeze(1), dropout_p=0.0)
    attn = attn.transpose(1, 2).reshape(B, T, layer.dim)
    x = x + layer.scale1 * F.linear(attn, layer.out_proj_weight)
    h = layer.norm2(x)
    h = F.linear(F.gelu(F.linear(h, layer.linear1)), layer.linear2)
    return x + layer.scale2 * h


def _init_deterministic(module: nn.Module, generator: torch.Generator) -> None:
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=generator))


# ===========================================================================
# 1. RoPE bit-exactness
# ===========================================================================


@pytest.mark.parametrize(
    "offset,T,head_dim",
    [(0, 1, 64), (5, 2, 64), (2999, 1, 128), (1, 4, 128), (250, 1, 64)],
)
def test_apply_rope_matches_legacy(offset, T, head_dim):
    gen = torch.Generator().manual_seed(0)
    B, H = 2, 3
    q = torch.randn(B, H, T, head_dim, generator=gen)
    k = torch.randn(B, H, T, head_dim, generator=gen)

    offset_t = torch.tensor([offset], dtype=torch.long)
    ref_q, ref_k = _legacy_apply_rope(q.clone(), k.clone(), offset_t)
    # `_rope_tables`'s second arg is only probed for `.shape[1]` (T) and
    # `.device` -- it takes the frame tensor `[B, T, dim]` in real callers,
    # not `q`/`k` (which are `[B, H, T, head_dim]` post fused-qkv split).
    frame_like = torch.empty(1, T, 1, device=q.device)
    rotr, roti = _rope_tables(head_dim, frame_like, offset_t)
    new_q, new_k = _apply_rope(q.clone(), k.clone(), rotr, roti)

    assert torch.equal(ref_q, new_q)
    assert torch.equal(ref_k, new_k)


def test_rope_tables_reused_across_layers_within_step():
    """One (offset, T, head_dim) table, computed once and applied to several
    different q/k pairs (as every layer in one step does), must still match
    the always-recompute-per-call oracle for each of them."""
    gen = torch.Generator().manual_seed(1)
    head_dim = 64
    offset_t = torch.tensor([42], dtype=torch.long)
    frame_like = torch.empty(1, 1, 1)  # T=1, matches the q/k built below
    rotr, roti = _rope_tables(head_dim, frame_like, offset_t)
    for _ in range(5):
        q = torch.randn(2, 3, 1, head_dim, generator=gen)
        k = torch.randn(2, 3, 1, head_dim, generator=gen)
        ref_q, ref_k = _legacy_apply_rope(q.clone(), k.clone(), offset_t)
        new_q, new_k = _apply_rope(q.clone(), k.clone(), rotr, roti)
        assert torch.equal(ref_q, new_q)
        assert torch.equal(ref_k, new_k)


# ===========================================================================
# 2. RingKV bit-exactness across a ring wrap, with elastic recycle interleaved
# ===========================================================================


def test_ringkv_complete_matches_legacy_across_wrap():
    gen = torch.Generator().manual_seed(2)
    B, H, D, capacity = 2, 1, 4, 5
    legacy = _LegacyRingKV(B, H, D, capacity, torch.device("cpu"), torch.float32)
    new = _RingKV(B, H, D, capacity, torch.device("cpu"), torch.float32)

    offset = torch.zeros(1, dtype=torch.long)
    num_steps = 4000  # capacity=5 -> hundreds of wraps
    for step in range(num_steps):
        T = 1
        k = torch.randn(B, H, T, D, generator=gen)
        v = torch.randn(B, H, T, D, generator=gen)

        legacy_keys, legacy_values, legacy_pos = legacy.complete(k.clone(), v.clone())
        indexes, positions_pre_mask = _ringkv_positions(offset, T, capacity, torch.device("cpu"))
        new_keys, new_values, new_pos = new.complete(k.clone(), v.clone(), indexes, positions_pre_mask)

        assert torch.equal(legacy_keys, new_keys), f"step {step}: cached keys mismatch"
        assert torch.equal(legacy_values, new_values), f"step {step}: cached values mismatch"
        assert torch.equal(legacy_pos, new_pos), f"step {step}: positions mismatch"
        offset = offset + T

        if step % 37 == 0:
            row = step % B
            legacy.reset_slot(row)
            new.reset_slot(row)
        if step % 53 == 0:
            row = (step + 1) % B
            legacy.bump_slot_start(row)
            new.bump_slot_start(row)

    assert torch.equal(legacy.end_offset, new.end_offset)
    assert torch.equal(legacy.start_offset, new.start_offset)
    assert torch.equal(legacy.cache, new.cache)


# ===========================================================================
# 3. End-to-end step(): Helium temporal stack
# ===========================================================================


def test_temporal_streaming_step_matches_legacy_end_to_end():
    dim, num_heads, hidden, num_layers, context, text_card = 32, 4, 64, 3, 16, 17
    B = 2

    real_stack = PersonaPlexTemporalStreaming(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden=hidden,
        context=context,
        text_card=text_card,
    )
    _init_deterministic(real_stack, torch.Generator().manual_seed(3))
    real_stack.eval()
    real_stack.streaming_init(batch_size=B)

    legacy_kvs = [
        _LegacyRingKV(B, num_heads, dim // num_heads, context, torch.device("cpu"), torch.float32)
        for _ in range(num_layers)
    ]
    legacy_offset = torch.zeros(1, dtype=torch.long)

    gen = torch.Generator().manual_seed(4)
    num_steps = 2000  # context=16 -> ~125 wraps
    for step_idx in range(num_steps):
        frame = torch.randn(B, 1, dim, generator=gen)

        x = frame.clone()
        for layer, kv in zip(real_stack.layers, legacy_kvs):
            x = _legacy_temporal_layer_forward(layer, x, kv, legacy_offset, context)
        legacy_offset = legacy_offset + x.shape[1]
        legacy_out = _rms_norm_f32(x, real_stack.out_norm_alpha, 1e-8)
        legacy_text_logits = F.linear(legacy_out, real_stack.text_linear)[:, None]

        new_out, new_text_logits = real_stack.step(frame.clone())

        assert torch.equal(legacy_out, new_out), f"step {step_idx}: transformer_out mismatch"
        assert torch.equal(legacy_text_logits, new_text_logits), f"step {step_idx}: text_logits mismatch"

        if step_idx % 71 == 0:
            row = step_idx % B
            for kv in legacy_kvs:
                kv.reset_slot(row)
            real_stack.reset_slot(row)
        if step_idx % 97 == 0:
            row = (step_idx + 1) % B
            for kv in legacy_kvs:
                kv.bump_slot_start(row)
            real_stack.bump_slot_start(row)

    for legacy_kv, real_kv in zip(legacy_kvs, real_stack._kv):
        assert torch.equal(legacy_kv.end_offset, real_kv.end_offset)
        assert torch.equal(legacy_kv.start_offset, real_kv.start_offset)


# ===========================================================================
# 4. End-to-end step(): Mimi encoder/decoder transformer
# ===========================================================================


def test_mimi_streaming_step_matches_legacy_end_to_end():
    dim, num_heads, num_layers, context = 16, 2, 2, 8
    B = 2

    real_stack = _MimiStreamingTransformer(num_layers=num_layers, dim=dim, num_heads=num_heads, context=context)
    _init_deterministic(real_stack, torch.Generator().manual_seed(5))
    real_stack.eval()
    real_stack.streaming_init(batch_size=B)

    legacy_kvs = [
        _LegacyRingKV(B, num_heads, dim // num_heads, context, torch.device("cpu"), torch.float32)
        for _ in range(num_layers)
    ]
    legacy_offset = torch.zeros(1, dtype=torch.long)

    gen = torch.Generator().manual_seed(6)
    num_steps = 1000  # context=8 -> ~125 wraps
    for step_idx in range(num_steps):
        frame = torch.randn(B, 1, dim, generator=gen)

        x = frame.clone()
        for layer, kv in zip(real_stack.layers, legacy_kvs):
            x = _legacy_mimi_layer_forward(layer, x, kv, legacy_offset, context)
        legacy_offset = legacy_offset + x.shape[1]

        new_x = real_stack.step(frame.clone())

        assert torch.equal(x, new_x), f"step {step_idx}: mimi transformer output mismatch"

        if step_idx % 41 == 0:
            row = step_idx % B
            for kv in legacy_kvs:
                kv.reset_slot(row)
            real_stack.reset_slot(row)

    for legacy_kv, real_kv in zip(legacy_kvs, real_stack._kv):
        assert torch.equal(legacy_kv.end_offset, real_kv.end_offset)
        assert torch.equal(legacy_kv.start_offset, real_kv.start_offset)
