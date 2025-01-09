#include <iostream>
#include <cuda.h>
#include <mma.h>
#include <stdlib.h>

#define INPUT_TYPE uint8_t
#define SCALE_TYPE uint8_t
#define ACCUM_TYPE int

#define WARP_SIZE 32
#define BLOCK_SIZE 16
#define TILE_DIM 16

using namespace nvcuda::wmma;

__global__ void tensorCoreMultiplyScaleAndAccumulate(
    const INPUT_TYPE *a, const INPUT_TYPE *b, ACCUM_TYPE *c, 
    const SCALE_TYPE *a_scale, const SCALE_TYPE *b_scale,
    int M, int K, int N) {
    // tid
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate global row and column for this thread block
    int wid_x = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int wid_y = (blockIdx.y * blockDim.y + threadIdx.y);

    if (wid_x >= N / 16 || wid_y >= M / 16) return;

    // Initialize fragments
    fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, INPUT_TYPE, row_major> a_frag;
    fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, INPUT_TYPE, col_major> b_frag;
    fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, ACCUM_TYPE> acc_frag;
    fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, ACCUM_TYPE> intermediate_frag;
    fill_fragment(acc_frag, 0);
    
    // #pragma unroll 1
    for (int k = 0; k < K; k += TILE_DIM) {
        // Initialize intermediate_frag
        fill_fragment(intermediate_frag, 0);

        // Load
        load_matrix_sync(a_frag, a + (wid_y * TILE_DIM) * K + k, K);
        load_matrix_sync(b_frag, b + (wid_x * TILE_DIM) * K + k, K);

        // Matrix Multiplication
        mma_sync(intermediate_frag, a_frag, b_frag, intermediate_frag);

        // Apply Scale and Accumulate
#if BLOCK_SIZE == 16
        int a_scale_idx = wid_y * 8 * (K / 16) + k/16 + (threadIdx.x % 32 / 16) * (K / 16);
        int b_scale_idx = wid_x * 8 * (K / 16) + k/16 + (threadIdx.x % 4 * 2) * (K / 16);
        acc_frag.x[0] += intermediate_frag.x[0] * a_scale[a_scale_idx] * b_scale[b_scale_idx];
        acc_frag.x[1] += intermediate_frag.x[1] * a_scale[a_scale_idx] * b_scale[b_scale_idx + 1];
        acc_frag.x[2] += intermediate_frag.x[2] * a_scale[a_scale_idx+4*(K/16)] * b_scale[b_scale_idx];
        acc_frag.x[3] += intermediate_frag.x[3] * a_scale[a_scale_idx+4*(K/16)] * b_scale[b_scale_idx + 1];
        acc_frag.x[4] += intermediate_frag.x[4] * a_scale[a_scale_idx] * b_scale[b_scale_idx+4*(K/16)];
        acc_frag.x[5] += intermediate_frag.x[5] * a_scale[a_scale_idx] * b_scale[b_scale_idx+4*(K/16) + 1];
        acc_frag.x[6] += intermediate_frag.x[6] * a_scale[a_scale_idx+4*(K/16)] * b_scale[b_scale_idx+4*(K/16)];
        acc_frag.x[7] += intermediate_frag.x[7] * a_scale[a_scale_idx+4*(K/16)] * b_scale[b_scale_idx+4*(K/16) + 1];
#elif BLOCK_SIZE == 32
        int a_scale_idx = wid_y * 8 * (K / 16) + k/16 + (threadIdx.x % 32 / 8) * (K / 16);
        int b_scale_idx = wid_x * 8 * (K / 16) + k/16 + (threadIdx.x % 4) * (K / 16);
        acc_frag.x[0] += intermediate_frag.x[0] * a_scale[a_scale_idx] * b_scale[b_scale_idx];
        acc_frag.x[1] += intermediate_frag.x[1] * a_scale[a_scale_idx] * b_scale[b_scale_idx];
        acc_frag.x[2] += intermediate_frag.x[2] * a_scale[a_scale_idx+4*(K/16)] * b_scale[b_scale_idx];
        acc_frag.x[3] += intermediate_frag.x[3] * a_scale[a_scale_idx+4*(K/16)] * b_scale[b_scale_idx];
        acc_frag.x[4] += intermediate_frag.x[4] * a_scale[a_scale_idx] * b_scale[b_scale_idx+4*(K/16)];
        acc_frag.x[5] += intermediate_frag.x[5] * a_scale[a_scale_idx] * b_scale[b_scale_idx+4*(K/16)];
        acc_frag.x[6] += intermediate_frag.x[6] * a_scale[a_scale_idx+4*(K/16)] * b_scale[b_scale_idx+4*(K/16)];
        acc_frag.x[7] += intermediate_frag.x[7] * a_scale[a_scale_idx+4*(K/16)] * b_scale[b_scale_idx+4*(K/16)];
#endif
        // acc_frag.x[0] += intermediate_frag.x[0];
        // acc_frag.x[1] += intermediate_frag.x[1];
        // acc_frag.x[2] += intermediate_frag.x[2];
        // acc_frag.x[3] += intermediate_frag.x[3];
        // acc_frag.x[4] += intermediate_frag.x[4];
        // acc_frag.x[5] += intermediate_frag.x[5];
        // acc_frag.x[6] += intermediate_frag.x[6];
        // acc_frag.x[7] += intermediate_frag.x[7];

        // For debugging
        // if (blockIdx.x == 0 && threadIdx.x == 1 && blockIdx.y == 0 && threadIdx.y == 0) {
        //     printf("Debugging\n");
        //     for (int i = 0; i < 16; i++) {
        //         printf("%d ", a_scale[i]);
        //     }
        //     printf("\n");
        //     printf("%d %d: %d %d, %d %d %d %d\n", wid_x, wid_y, a_scale_idx, b_scale_idx, a_scale[a_scale_idx], a_scale[a_scale_idx+4*(K/16)], b_scale[b_scale_idx], b_scale[b_scale_idx+4*(K/16)]);
        // }
    }

    // Store
    store_matrix_sync(c + (wid_y * TILE_DIM) * N + wid_x * TILE_DIM, acc_frag, N, mem_row_major);
}

int main(void)
{
    // int M = 16; // Example dimensions, must be multiples of TILE_DIM
    // int K = 32;
    // int N = 16;

    int M = 6272; // Example dimensions, must be multiples of TILE_DIM
    int K = 768;
    int N = 1280;

    // CPU Allocation
    INPUT_TYPE* a = (INPUT_TYPE*)malloc(M * K * sizeof(INPUT_TYPE));
    INPUT_TYPE* b = (INPUT_TYPE*)malloc(K * N * sizeof(INPUT_TYPE));
    ACCUM_TYPE* c = (ACCUM_TYPE*)malloc(M * N * sizeof(ACCUM_TYPE));
    SCALE_TYPE* a_scale = (SCALE_TYPE*)malloc((M*K)/BLOCK_SIZE * sizeof(SCALE_TYPE));
    SCALE_TYPE* b_scale = (SCALE_TYPE*)malloc((K*N)/BLOCK_SIZE * sizeof(SCALE_TYPE));

    // Initialization
    for (int i = 0; i < M; i++) for (int j = 0; j < K; j++) {
        a[i * K + j] = i * K + j;
    }

    for (int i = 0; i < K; i++) for (int j = 0; j < N; j++) {
        b[i * N + j] = i * N + j;
    }

    for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
        c[i * N + j] = 0;
    }

    for (int i = 0; i < (M*K)/BLOCK_SIZE; i++) {
        a_scale[i] = i;
    }

    for (int i = 0; i < (K*N)/BLOCK_SIZE; i++) {
        b_scale[i] = i;
    }

    // GPU Allocation
    INPUT_TYPE* d_a;
    INPUT_TYPE* d_b;
    ACCUM_TYPE* d_c;
    SCALE_TYPE* d_a_scale;
    SCALE_TYPE* d_b_scale;

    cudaMalloc((void**)&d_a, M * K * sizeof(INPUT_TYPE));
    cudaMalloc((void**)&d_b, K * N * sizeof(INPUT_TYPE));
    cudaMalloc((void**)&d_c, M * N * sizeof(ACCUM_TYPE));
    cudaMalloc((void**)&d_a_scale, (M*K)/BLOCK_SIZE * sizeof(SCALE_TYPE));
    cudaMalloc((void**)&d_b_scale, (K*N)/BLOCK_SIZE * sizeof(SCALE_TYPE));

    // GPU Memcpy
    cudaMemcpy(d_a, a, M * K * sizeof(INPUT_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, K * N * sizeof(INPUT_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, M * N * sizeof(ACCUM_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_scale, a_scale, (M*K)/BLOCK_SIZE * sizeof(SCALE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_scale, b_scale, (K*N)/BLOCK_SIZE * sizeof(SCALE_TYPE), cudaMemcpyHostToDevice);

    // Launch kernel with enough threads to cover the fragments
    dim3 blockDim(WARP_SIZE * 4, 4);
    dim3 gridDim(max(N / (TILE_DIM * 4), 1), max(M / (TILE_DIM * 4), 1));

    tensorCoreMultiplyScaleAndAccumulate<<<gridDim, blockDim>>>(d_a, d_b, d_c, d_a_scale, d_b_scale, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, M * N * sizeof(ACCUM_TYPE), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < K; j++) {
    //         std::cout << int(a[i * N + j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < K; i++) {
    //     for (int j = 0; j < M; j++) {
    //         std::cout << int(b[i * N + j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            std::cout << c[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
