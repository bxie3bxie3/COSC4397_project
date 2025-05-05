#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script implements and benchmarks a simple matrix multiplication (GEMM) using Triton.
It measures execution time and calculates throughput in GFLOPS.
"""

import torch
import triton
import triton.language as tl
import sys
import traceback

@triton.jit
def matmul_kernel(
    A, B, C,  # pointers to input A, B and output C
    M, N, K,  # matrix dimensions
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for i in range(0, K, BLOCK_K):
        offs_k = i + tl.arange(0, BLOCK_K)

        a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        
        accumulator += tl.dot(a, b)

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)

def matmul_triton(a: torch.Tensor, b: torch.Tensor, block_size=(128, 128, 32)) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Inputs must be CUDA tensors.")
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "K dimensions must match"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    BM, BN, BK = block_size
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BM, BN, BK,
    )
    return c

def run_benchmark(M, N, K, dtype=torch.float32):
    print(f"\n--- GEMM Benchmark ({M}x{K} * {K}x{N}) ---")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)

    for _ in range(10):
        _ = matmul_triton(a, b)
    torch.cuda.synchronize()

    # time Triton
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    c_triton = matmul_triton(a, b)
    end.record()
    torch.cuda.synchronize()
    t_triton = start.elapsed_time(end)  # ms

    # time PyTorch
    for _ in range(10):
        _ = a @ b
    torch.cuda.synchronize()
    start.record()
    c_torch = a @ b
    end.record()
    torch.cuda.synchronize()
    t_torch = start.elapsed_time(end)

    c_torch = a @ b
    c_triton = matmul_triton(a, b)
    diff = (c_triton - c_torch).abs()
    print("min/max/mean of mismatch:", diff.min().item(), "/", 
            diff.max().item(), "/", diff.mean().item())

    if not torch.allclose(c_triton, c_torch, rtol=3e-3, atol = 1e-1):
        print("Mismatch!")

    # compute GFLOPS
    ops = 2 * M * N * K
    gflops_triton = ops / (t_triton / 1e3) / 1e9
    gflops_torch = ops / (t_torch / 1e3) / 1e9

#    print(f"Triton: {t_triton:.2f} ms, {gflops_triton:.1f} GFLOPS")
    print(f"Triton: {gflops_triton:.1f} GFLOPS")
#    print(f"PyTorch: {t_torch:.2f} ms, {gflops_torch:.1f} GFLOPS")
    print(f"PyTorch: {gflops_torch:.1f} GFLOPS")
# --- Main ---
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required.")
        sys.exit(1)
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # test sizes
test_sizes = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]
for M, N, K in test_sizes:
    run_benchmark(M, N, K)

