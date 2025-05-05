#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// A: m x k, B: k x n, C: m x n (row-major)

// Kernel 1: naive row-indexing
__global__ void matmul1_naive(float *A, float *B, float *C, int M, int N, int K) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // row index
    int j = threadIdx.y + blockIdx.y * blockDim.y; // column index
    if (i >= M || j >= N) return; 
    float c = 0.0f;
    for (int k = 0; k < K; k++) {
        c += A[i*K + k] * B[k*N + j];
    }
    C[i*N + j] = c;
}

// Kernel 2: coalesced indexing (swapped thread indices)
__global__ void matmul2_coalesced(float *A, float *B, float *C, int M, int N, int K) {
    int j = threadIdx.x + blockIdx.x * blockDim.x; // column index
    int i = threadIdx.y + blockIdx.y * blockDim.y; // row index
    if (i >= M || j >= N) return; 
    float c = 0.0f;
    for (int k = 0; k < K; k++) {
        c += A[i*K + k] * B[k*N + j];
    }
    C[i*N + j] = c;
}

// kernel 3: coalesced, coarsened 2x2
// TODO: develop your code here based on Kernel 2.
// Add thread coarsening: start with 2x2, meaning each thread now is
// responsible to compute 4 elements (a 2x2 patch) of C.
// Example GFLOPS if successful: ~3000 GFLOPS for matrix size 4096
__global__ void coarsened_matmul2x2(float *A, float *B, float *C, int M, int N, int K)
{
	int j = 2 * (threadIdx.x + blockIdx.x * blockDim.x); // column index
    	int i = 2 * (threadIdx.y + blockIdx.y * blockDim.y); // row index
	float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
	for (int k = 0; k < K; k++) 
	{	
		float a0 = 0.0f, a1 = 0.0f, b0 = 0.0f, b1 = 0.0f;
		if (i < M)
			a0 = A[i * K + k];
		if ( i + 1 < M)
			a1 = A[(i + 1) * K + k];
        	if (j < N)
			b0 = B[k * N + j];
		if (j + 1 < N)
			b1 = B[k * N + j + 1];
		if ( i < M &&  j < N)
			c00 += a0 * b0;
		if (i + 1 < M && j < N)
			c01 += a0 * b1;
		if (i < M && j + 1 < N)
			c10 += a1 * b0;
		if (i + 1 < M && j + 1 < N)
			c11 += a1 * b1;
    	}
	if (i < M && j < N)
		C[i * N + j] = c00;
	if (i + 1 < M && j < N)
		C[i * N + j + 1] = c01;
	if (i < M && j + 1 < N)
	       	C[(i + 1) * N + j] = c10;
	if (i + 1 < M && j + 1 < N)
		C[(i + 1) * N + j + 1] = c11;
}

// Kernel 4: shared memory (SMEM) tiled
// TODO: develop your code here. You should make use of shared memory
// and Tiling technique to reduce global memory access. 
// Each thread block computes a TSxTS block of C, where
// A is MxK, B is KxN, and C is MxN (all row‑major).
// Each thread is still responsible to compute one element in C
#define TS 16
__global__ void MatMulTiled(float *A, float *B, float *C, int M, int N, int K)
{
	
	__shared__ float Adata[TS][TS];
	__shared__ float Bdata[TS][TS];

  	int j = threadIdx.x + blockIdx.x * blockDim.x; // column index
    	int i = threadIdx.y + blockIdx.y * blockDim.y; // row index 
    	
	float c = 0.0f;
    	for (int k = 0; k < K / TS; ++k) 
	{
		if (i < M && (k * TS + threadIdx.x) < K)
			Adata[threadIdx.y][threadIdx.x] 
				= A[i * K + k * TS + threadIdx.x];
		else
			Adata[threadIdx.y][threadIdx.x] = 0.0f;
		if ((k * TS + threadIdx.y) < K && j < N)
			Bdata[threadIdx.y][threadIdx.x] 
				= B[(k * TS + threadIdx.y) * N + j];		
	      	else
			Bdata[threadIdx.y][threadIdx.x] = 0.0f;	
		__syncthreads();
        	for (int kk = 0; kk < TS; ++kk)
		{
			c += Adata[threadIdx.y][kk] * Bdata[kk][threadIdx.x];
		}
		__syncthreads();
	}
	if (i < M && j < N)
		C[i * N + j] = c; 
}

// Kernel 5: shared memory (SMEM) tiled
// TODO: Do your best to have the fastest correct kernel here.
// Things to do (might be a good idea in this order):
// 1. add thread coarsening to previous tiling kernel: for example,
//    your thread block dim can be 16x16 (256 threads per block), yet
//    it computes a 32x32 matrix block in C. Each thread computes 2x2 elements
//    in C.
//    Example GFLOPS: ~4000 GFLOPS. 
// 2. increase the coarsening factor to 4x4, 8x8 etc.
//    Example GFLOPS: 4x4 ~8000 GFLOPS, 8x8 ~10000 GFLOPS
// 3. Tuning parameters: block dim, coarsening factor, ... 
#define BS 16   
__global__ void MatmulBest(float *A, float *B, float *C, int M, int N, int K)
{
	__shared__ float Adata[4 * BS][BS],
			Bdata[BS][4 * BS];

	int by = 4 * blockIdx.y * BS,
	    bx = 4 * blockIdx.x * BS, 
	    ty = threadIdx.y,
	    tx = threadIdx.x;

	float c00 = 0.0f, 
	      c01 = 0.0f, 
	      c02 = 0.0f,
	      c03 = 0.0f, 
	      c10 = 0.0f,
	      c11 = 0.0f,
	      c12 = 0.0f,
	      c13 = 0.0f, 
	      c20 = 0.0f,
	      c21 = 0.0f,
	      c22 = 0.0f, 
	      c23 = 0.0f, 
	      c30 = 0.0f, 
	      c31 = 0.0f, 
	      c32 = 0.0f, 
	      c33 = 0.0f;

	for (int k = 0; k < K / BS; ++k)
	{
		if ((by + 4 * ty) < M && (k * BS + tx) < K)
		       	Adata[4 * ty][tx] 
				= A[(by + 4 * ty) * K + (k * BS + tx)];
		else
			Adata[4 * ty][tx] = 0.0f;
		if ((by + 4 * ty + 1) < M && (k * BS + tx) < K)
			Adata[4 * ty + 1][tx] 
				= A[(by + 4 * ty + 1) * K + (k * BS + tx)] ;
		else
			Adata[4 * ty + 1][tx] = 0.0f;
		if ((by + 4 * ty + 2) < M && (k * BS + tx) < K)
			Adata[4 * ty + 2][tx]
				= A[(by + 4 * ty + 2) * K + (k * BS + tx)];
		else
			Adata[4 * ty + 2][tx] = 0.0f;
		if ((by + 4 * ty + 3) < M && (k * BS + tx) < K)
			Adata[4 * ty + 3][tx]
				= A[(by + 4 * ty + 3) * K + (k * BS + tx)];
		else
			Adata[4 * ty + 3][tx] = 0.0f;

		if ((k * BS + ty) < K && (bx + 4 * tx) < N)
			Bdata[ty][4 * tx] 
				= B[(k * BS + ty)* N + (bx + 4 * tx)]; 
		else
 			Bdata[ty][4 * tx] = 0.0f;
		if ((k * BS + ty) < K && (bx + 4 * tx + 1) < N)
			Bdata[ty][4 * tx + 1] 
				= B[(k * BS + ty) * N + (bx + 4 * tx + 1)]; 		
		else
			Bdata[ty][4 * tx + 1] = 0.0f;
		if ((k * BS + ty) < K && (bx + 4 * tx + 2) < N)
			Bdata[ty][4 * tx + 2] 
				= B[(k * BS + ty) * N + (bx + 4 * tx + 2)]; 		
		else
			Bdata[ty][4 * tx + 2] = 0.0f;
		if ((k * BS + ty) < K && (bx + 4 * tx + 3) < N)
			Bdata[ty][4 * tx + 3] 
				= B[(k * BS + ty) * N + (bx + 4 * tx + 3)]; 		
		else
			Bdata[ty][4 * tx + 3] = 0.0f;
		__syncthreads();
		for (int kk = 0; kk < BS; ++kk)
		{
			float a0 = Adata[4 * ty][kk],
			      a1 = Adata[4 * ty + 1][kk],
			      a2 = Adata[4 * ty + 2][kk],
			      a3 = Adata[4 * ty + 3][kk],
			      b0 = Bdata[kk][4 * tx],
			      b1 = Bdata[kk][4 * tx + 1],
			      b2 = Bdata[kk][4 * tx + 2],
			      b3 = Bdata[kk][4 * tx + 3];
			c00 += a0 * b0;
		       	c01 += a0 * b1;
			c02 += a0 * b2;
			c03 += a0 * b3;
			c10 += a1 * b0;
			c11 += a1 * b1;
			c12 += a1 * b2;
			c13 += a1 * b3;
			c20 += a2 * b0;
			c21 += a2 * b1;
			c22 += a2 * b2;
			c23 += a2 * b3;
			c30 += a3 * b0;
			c31 += a3 * b1;
			c32 += a3 * b2;
			c33 += a3 * b3;
		
		}
		__syncthreads();
	}

	int row = by + 4 * ty,
	    col = bx + 4 * tx;

		if (row < M && col < N)
			C[row * N + col] = c00;
		if (row < M && col + 1 < N)
			C[row * N + col + 1] = c01;
		if (row < M && col + 2 < N)
			C[row * N + col + 2] = c02;
		if (row < M && col + 3 < N)
			C[row * N + col + 3] = c03;
		if (row + 1 < M && col < N)
			C[(row + 1) * N + col] = c10;
		if (row + 1 < M && col + 1 < N)
			C[(row + 1) * N + col + 1] = c11;

		if (row + 1 < M && col + 2 < N)
			C[(row + 1) * N + col + 2] = c12;

		if (row + 1 < M && col + 3 < N)
			C[(row + 1) * N + col + 3] = c13;

		if (row + 2 < M && col < N)
			C[(row + 2) * N + col] = c20;
		if (row + 2 < M && col + 1 < N)
			C[(row + 2) * N + col+ 1] = c21;
		if (row + 2 < M && col + 2 < N)
			C[(row + 2) * N + col + 2] = c22;
		if (row + 2 < M && col + 3 < N)
			C[(row + 2) * N + col + 3] = c23;
		if (row + 3 < M && col < N)
			C[(row + 3) * N + col] = c30;
		if (row + 3 < M && col + 1 < N)
			C[(row + 3) * N + col + 1] = c31;
		if (row + 3 < M && col + 2 < N)
			C[(row + 3) * N + col + 2] = c32;
		if (row + 3 < M && col + 3 < N)
			C[(row + 3) * N + col + 3] = c33;


}



//------------------------------------------------------------------------------
// Kernel launcher function
//
// This function selects which kernel to run based on kernelId.
// Initially each kernel is implemented in a “naive” (correct but slow) way.
// Students are meant to modify/fill in the kernels and tune the launch parameters.
extern "C" void launchMatMulKernel(int kernelId, float *A, float *B, float *C, int M, int N, int K) {
    // For demonstration, we use kernelId values 1-4.
    // (Other values can be added as more kernels are implemented.)
    if (kernelId == 1) {
        // Naive kernel: one thread per output element.
        dim3 blockDim(16, 16);
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                     (N + blockDim.y - 1) / blockDim.y);
        matmul1_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 2) {
        // Coalesced indexing version.
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);
        matmul2_coalesced<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 3) {
        // 2x2 coarsened kernel.
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 31) / 32, (M + 31) / 32);
        coarsened_matmul2x2<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 4) {
        // Shared memory tiled version.
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 15) / 16, (M + 15) / 16);
        MatMulTiled<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else if (kernelId == 5) {
	// Shared memory tiled version.
        dim3 blockDim(16, 16);
        dim3 gridDim((N + 63) / 64, 
			(M + 63) / 64);
        MatmulBest<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    }
    else {
        // Default: if an unknown kernelId is passed, run the naive version.
        dim3 blockDim(16, 16);
        dim3 gridDim((M + 31) / 32,
                     (N + 31) / 32);
        matmul1_naive<<<gridDim, blockDim>>>(A, B, C, M, N, K);
    
   }
    // Make sure the kernel has finished.
    cudaDeviceSynchronize();
}
