#include <iostream>
#include <stdint.h>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"
#include <mma.h>
using namespace nvcuda;

#define errCheck(ans) { checkError((ans), __FILE__, __LINE__); }
inline void checkError(cudaError_t err, const char * file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s --> %s:%d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

#define TILE_WIDTH_16 16
#define MAX_NUM_THREADS 1024
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;


__global__ void conv_forward_kernel(__half *X_unroll, __half *X, __half *null,
     const int Batch, const int Map_out, const int C, const int H, const int W, const int K)
{
    int c,s,h_out, w_out, p,q, w_unroll, w_base, h_unroll;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    #define X_unroll( i1, i0) X_unroll[ (i1) * (W_unroll) + (i0)]
    #define X(i2, i1, i0) X[(i2) * (H * W) + (i1) * (W) + (i0)]

    if (t < (C * W_unroll)) {
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / W_out;
        w_out = s % W_out;
        w_unroll = h_out * W_out + w_out;
        w_base = c * K * K;
        for(p = 0; p < K; p++){
            for(q = 0; q < K; q++){
                h_unroll = w_base + p * K + q;
                X_unroll((h_unroll), (w_unroll)) = X((c), (h_out + p), (w_out + q)) ;
            }
        }
    }
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__global__ void matrixMultiply(__half *mask, __half *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns, int origm, int orign)
{

    __shared__ __half subTileM[TILE_WIDTH_16][TILE_WIDTH_16];
    __shared__ __half subTileN[TILE_WIDTH_16][TILE_WIDTH_16];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // num of tiles in A and B both directions
    int num_tiles = ceil((1.0*numBRows)/TILE_WIDTH_16);

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH_16 + ty;
    int Col = bx * TILE_WIDTH_16 + tx;
    __half Pvalue = 0.0;

    Pvalue=0.0;
    for (int q = 0; q < num_tiles; ++q) {
        // Collaborative loading of M and N tiles into shared memory
        int col = q* TILE_WIDTH_16 + tx;
        int row = q * TILE_WIDTH_16 + ty;
        if(Row < numCRows) {
            if(col < numBRows ) subTileM[ty][tx] = mask[Row*numAColumns + col];
            else subTileM[ty][tx] = 0.0;
        }
        else{
            subTileM[ty][tx] = 0.0;
        }
        if(Col < numCColumns) {
            if(row < numBRows ) subTileN[ty][tx] = B[row*numBColumns + Col];
            else subTileN[ty][tx] = 0.0;
        }
        else{
            subTileN[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < 16; ++k)
                Pvalue += subTileM[ty][k] * subTileN[k][tx];

        __syncthreads();
    }
     if(Row < origm && Col < orign) C[Row*orign+Col] = __half2float(Pvalue);
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int W = Width;
    int H = Height;
    int C= Channel;
    int W_out = W - K + 1;
    int H_out = H - K + 1;
    int H_unroll = C * K * K;
    int W_unroll = H_out * W_out;

    int inputSize = Batch * Channel * Height * Width;
    int maskSize = Map_out * Channel * K * K;
    int outputSize = Batch * Map_out * H_out * W_out;

     std::cout<<"Height_out: "<< H_out <<std::endl;
    std::cout<<"Width_out: "<< W_out <<std::endl;
    std::cout<<"Channel: "<< Channel <<std::endl;
    std::cout<<"K: "<< K <<std::endl;
    std::cout<<"Map_out: "<< Map_out <<std::endl;

    __half *device_X_unrolled, *device_X;
    __half *device_mask_padded, *device_X_unrolled_padded;
    float *device_output;
    errCheck(cudaMalloc((void **) &device_X_unrolled, W_unroll * H_unroll * sizeof(__half)));
    errCheck(cudaMalloc((void **) &device_X, inputSize * sizeof(__half)));
    errCheck(cudaMalloc((void **) &device_output, (outputSize/Batch) * sizeof(float)));


    __half *h_host_input, *h_host_mask, *h_host_paddedX, *h_host_paddedmask;
    // transform inputs to half.
    h_host_mask = (__half*) malloc(maskSize*sizeof(__half));
    for (int i=0; i<maskSize; i++){
        h_host_mask[i] = __float2half(host_mask[i]);
    }
    h_host_input = (__half*) malloc(inputSize*sizeof(__half));
    for (int i=0; i<inputSize; i++){
        h_host_input[i] = __float2half(host_input[i]);
    }
    // load half arrays to unroll
    errCheck(cudaMemcpy(device_X, h_host_input, inputSize * sizeof(__half), cudaMemcpyHostToDevice));

    __half *h_host_tempunrolled;
    int padded_m = Map_out + (ceil(Map_out/16.0) - Map_out/16.0)*16.0;
    int padded_k = H_unroll + (ceil(H_unroll/16.0) - H_unroll/16.0)*16.0;
    int padded_n = W_unroll + (ceil(W_unroll/16.0) - W_unroll/16.0)*16.0;
    h_host_paddedmask = (__half*) malloc(padded_k * padded_m * sizeof(__half));
    h_host_paddedX = (__half*) malloc(padded_k * padded_n * sizeof(__half));
    h_host_tempunrolled = (__half*) malloc(W_unroll * H_unroll * sizeof(__half));
    errCheck(cudaMalloc((void **) &device_X_unrolled_padded, padded_k * padded_n * sizeof(__half)));
    errCheck(cudaMalloc((void **) &device_mask_padded, padded_k * padded_m * sizeof(__half)));

    int num_threads = C * H_out * W_out;
    int num_blocks = ceil((1.0 * num_threads) / MAX_NUM_THREADS);

    dim3 gridDim (ceil(1.0 * padded_n / TILE_WIDTH_16),  ceil(1.0 *  padded_m/ TILE_WIDTH_16), 1);
    dim3 blockDim (TILE_WIDTH_16, TILE_WIDTH_16, 1);

    for (int n=0; n < Batch; n++) {
        // unroll kernel
        conv_forward_kernel<<<num_blocks, MAX_NUM_THREADS>>>(device_X_unrolled, device_X + n*(inputSize/Batch), NULL,Batch,Map_out,C,  H,  W,  K);
        errCheck(cudaMemcpy(h_host_tempunrolled, device_X_unrolled, (W_unroll * H_unroll) * sizeof(__half), cudaMemcpyDeviceToHost));

        for(int i=0; i<padded_k; i++){
            for(int j=0; j<padded_n; j++){
                if(i<H_unroll && j<W_unroll) h_host_paddedX[i*padded_n +j] = h_host_tempunrolled[i*W_unroll +j];
                else h_host_paddedX[i*padded_n +j] = __float2half(0.0);
            }
        }
        for(int i=0; i<padded_m; i++){
            for(int j=0; j<padded_k; j++){
                if(i<Map_out && j<H_unroll) h_host_paddedmask[i*padded_k +j] = h_host_mask[i*H_unroll +j];
                else h_host_paddedmask[i*padded_k +j] = __float2half(0.0);
            }
        }

        //std::cout << "padded arr for unrolled inside: " << __half2float(h_host_paddedX[(H_unroll-3)*padded_n+(W_unroll-1)]) << std::endl;
        //std::cout << "padded arr for unrolled outside: " << __half2float(h_host_paddedX[(H_unroll+3)*padded_n+(W_unroll-1)]) << std::endl;
        errCheck(cudaMemcpy(device_X_unrolled_padded, h_host_paddedX, padded_k * padded_n * sizeof(__half), cudaMemcpyHostToDevice));
        errCheck(cudaMemcpy(device_mask_padded, h_host_paddedmask, padded_k * padded_m * sizeof(__half), cudaMemcpyHostToDevice));

        matrixMultiply<<<gridDim, blockDim>>>(device_mask_padded, device_X_unrolled_padded, device_output, padded_m,
                               padded_k, padded_k,
                               padded_n,
                               padded_m,padded_n,
                               Map_out, W_unroll ) ;

        errCheck(cudaMemcpy((void *) (host_output + n*(outputSize/Batch)), device_output, (outputSize/Batch) * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Free device memory
    errCheck(cudaFree(device_output));
    errCheck(cudaFree(device_X_unrolled));
    errCheck(cudaFree(device_X));

    errCheck(cudaFree(device_mask_padded));
    errCheck(cudaFree(device_X_unrolled_padded));


    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
         std::cout<<"CUDA all error: "<<cudaGetErrorString(error)<<std::endl;
         exit(-1);
     }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask,
      const int Batch, const int Map_out, const int Channel, const int Height,
      const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
         std::cout<<"CUDA kern error: "<<cudaGetErrorString(error)<<std::endl;
         exit(-1);
     }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host'

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
         std::cout<<"CUDA clean error: "<<cudaGetErrorString(error)<<std::endl;
         exit(-1);
     }
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}