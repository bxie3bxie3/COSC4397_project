import torch
import triton
import triton.language as tl
import sys
import traceback

@triton.jit
def matmul_kernel(
    A, B, C,  
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BS_M: tl.constexpr, BS_N: tl.constexpr, BS_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BS_M + tl.arange(0, BS_M)
    offs_n = pid_n * BS_N + tl.arange(0, BS_N)

    accumulator = tl.zeros((BS_M, BS_N), dtype=tl.float32)

    for i in range(0, K, BS_K):
        offs_k = i + tl.arange(0, BS_K)
        
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        block_a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        block_b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        accumulator += tl.dot(block_a, block_b)

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask_C = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask_C)

def matmul_triton(a: torch.Tensor, b: torch.Tensor, block_size=(64, 64, 16)) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Inputs must be CUDA tensors.")
    M, K1 = a.shape
    K2, N = b.shape
    assert K1 == K2, "K dimensions must match"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BM, BN, BK = block_size
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](
        a, b, c,
        M, N, K1,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM, BN, BK,
    )
    return c

def run_benchmark(M, N, K, provider = triton, dtype=torch.float32):
    print(f"\n--- GEMM Benchmark ({M}x{K} * {K}x{N}) ---")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return None, None
    a = torch.randn((M, K), device='cuda', dtype=dtype).contiguous()
    b = torch.randn((K, N), device='cuda', dtype=dtype).contiguous()
    
    if provider == 'triton':
        func = matmul_triton
    else:
        func = lambda x, y: (x @ y).contiguous()

    for _ in range(10):
        _ = func(a, b)
    torch.cuda.synchronize()
    
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: func(a, b), quantiles=quantiles, rep=200)
    
    total_ops = 2 * M * N * K
    if ms > 0:
        gflops = total_ops / (ms / 1e3) / 1e9
    else:
        gflops = float('inf')

    if provider == 'triton':
        torch.cuda.synchronize()
        c_triton = func(a, b)
        torch.cuda.synchronize()
        c_torch = (a @ b).contiguous()
        torch.cuda.synchronize()

        if not torch.allclose(c_triton, c_torch, atol=1e-1, rtol=5e-3):
            print(f"Verification FAILED: Triton result differs from PyTorch result for size ({M}x{K} x {K} x {N}).")
            diff = torch.abs(c_triton - c_torch)
            print(f"Max absolute difference: {diff.max().item()}")
            print(f"Mean absolute difference: {diff.mean().item()}")
            print(f"Min absolute difference: {diff.min().item()}")
        else:
            print("Verification PASSED")

    return ms, gflops 

def print_summary(results):
    print("\n--- Benchmark Summary ---")
    print(f"{'Size (MxNxK)':<15} | {'Provider':<10} | {'Median Time (ms)':<18} | {' (GFLOPS/s)':<18}")
    print("-" * 70)
    for (M, N, K), providers in results.items():
        for provider, metrics in providers.items():
            ms_str = f"{metrics['ms']:.3f}" if metrics and metrics.get('ms') is not None else "N/A"
            gflops_str = f"{metrics['gflops']:.2f}" if metrics and metrics.get('gflops') is not None else "N/A"
            print(f"{f'{M}x{N}x{K}':<15} | {provider:<10} | {ms_str:<18} | {gflops_str:<18}")
        print("-" * 70)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA device unavailable.")
        sys.exit(1)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Triton version: {triton.__version__}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"GPU: {device_name}")
    
    test_sizes = [
        (32, 32, 32),
        (1024, 1024, 1024),
        (1024, 1024, 2048),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    results = {}

    for M, N, K in test_sizes:
        results[(M, N, K)] = {}
        
        ms_triton, gflops_triton = run_benchmark(M, N, K, provider='triton')
        results[(M, N, K)]['triton'] = {'ms': ms_triton, 'gflops': gflops_triton}

        ms_torch, gflops_torch = run_benchmark(M, N, K, provider='pytorch')
        results[(M, N, K)]['pytorch'] = {'ms': ms_torch, 'gflops': gflops_torch}
    
    print_summary(results)

