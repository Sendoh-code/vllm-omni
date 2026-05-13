import torch
import torch.nn.functional as F
import triton
from vllm_omni.diffusion.layers.norm import LayerNorm
from sglangnorm import _layer_norm_fwd_1pass_kernel


SHAPES = [
    # (batch, seq_len, hidden_dim)
    (1,  4096,  1536),   # Wan2.2-1.3B
    (4,  4096,  1536),
    (1, 16384,  1536),
    (1,  4096,  5120),   # Wan2.2-14B
    (4,  4096,  5120),
]

WARMUP  = 50
REPEATS = 200
DTYPE   = torch.bfloat16
DEVICE  = "cuda"
EPS     = 1e-6

def bench(fn, warmup=WARMUP, repeats=REPEATS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
 
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats * 1e3

def triton_layer_norm(
    x:      torch.Tensor,           # (..., D)  any leading dims
    weight: torch.Tensor,           # (D,)
    bias:   torch.Tensor,           # (D,)
    eps:    float = 1e-6,
) -> torch.Tensor:
    """
    Drop-in replacement for vllm-omni LayerNorm.forward_native.
    Stays in bf16/fp16 throughout — no fp32 cast round-trip.
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])   # (M, N)
    M, N = x_2d.shape
 
    # outputs
    y    = torch.empty_like(x_2d)
    mean = torch.empty(M, dtype=torch.float32, device=x.device)
    rstd = torch.empty(M, dtype=torch.float32, device=x.device)
 
    BLOCK_N = triton.next_power_of_2(N)
    # Triton caps at 2^17 = 131072 elements per block; assert just in case
    assert BLOCK_N <= 131072, f"hidden_dim {N} too large for single-block kernel"
 
    grid = (M,)
    _layer_norm_fwd_1pass_kernel[grid](
        # data pointers
        x_2d, y, weight, bias,
        # unused optional pointers — pass x_2d as a harmless dummy
        x_2d, x_2d, weight, bias, y, x_2d,
        x_2d, x_2d, x_2d, x_2d,
        mean, rstd,
        # strides
        x_2d.stride(0), y.stride(0),
        x_2d.stride(0), x_2d.stride(0),
        x_2d.stride(0), y.stride(0),
        # scalars
        M, N, eps,
        0.0,    # dropout_p
        False,  # zero_centered_weight
        # constexprs — only the ones we actually need are True
        IS_RMS_NORM        = False,
        BLOCK_N            = BLOCK_N,
        HAS_RESIDUAL       = False,
        STORE_RESIDUAL_OUT = False,
        HAS_WEIGHT         = True,
        HAS_BIAS           = True,
        HAS_DROPOUT        = False,
        STORE_DROPOUT_MASK = False,
        HAS_ROWSCALE       = False,
        HAS_X1             = False,
        HAS_W1             = False,
        HAS_B1             = False,
    )
    return y.reshape(orig_shape)

def check(ref, out, atol=1e-2, rtol=1e-2):
    max_diff = (ref.float() - out.float()).abs().max().item()
    ok = torch.allclose(ref.float(), out.float(), atol=atol, rtol=rtol)
    return ok, max_diff

for B,S,D in SHAPES:
    
    print(f"\nShape  batch={B}  seq={S}  dim={D}")
    print("-" * 48)

    x  = torch.randn(B, S, D, dtype=DTYPE, device=DEVICE)
    ln = LayerNorm(D).to(DEVICE).to(DTYPE)

    def native_fn(x): return ln.forward_native(x)
    compiled_fn = torch.compile(native_fn)

    def cuda_fn(x): return ln.forward_cuda(x)

    # torch.compile
    t_compile = bench(lambda: compiled_fn(x))
    ref = compiled_fn(x)
    out_compile = compiled_fn(x)
    ok, md = check(ref, out_compile)
    print(f"  torch.compile(native)   : {t_compile:8.2f} µs  ")

    t_cuda   = bench(lambda: cuda_fn(x))
    out_cuda = cuda_fn(x)
    ok, md = check(ref, out_cuda)
    print(f"  cuda          : {t_cuda:8.2f} µs  "

          f"max_diff={md:.2e}  {'✅' if ok else '❌'}")

    # triton
    t_triton   = bench(lambda: triton_layer_norm(x, ln.weight, ln.bias, EPS))
    out_triton = triton_layer_norm(x, ln.weight, ln.bias, EPS)
    ok, md = check(ref, out_triton)
    print(f"  SGLang Triton           : {t_triton:8.2f} µs  "

          f"max_diff={md:.2e}  {'✅' if ok else '❌'}")

 





