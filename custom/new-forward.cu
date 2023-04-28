#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"


#define errCheck(ans) { checkError((ans), __FILE__, __LINE__); }
inline void checkError(cudaError_t err, const char * file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s --> %s:%d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

__constant__ float mask[3136];
#define TILE_WIDTH 16
#define MAX_NUM_THREADS 1024

/*
 * worse solution with 1d tiles
__global__ void matrixMultiply(float *mask, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns, int Channel, int K, int H_out, int W_out) {
    //@@ Insert code to implement matrix multiplication here

    int maskSize = K*K;
    int ImgW = H_out*W_out;
    int ImgBlockSize = K*K*H_out*W_out;
    __shared__ float subTileM[49];
    __shared__ float subTileN[49];

    int map = blockIdx.x; int imgCols = blockIdx.y;
    int tx = threadIdx.x;

    // num of tiles in A and B both directions
    float Pvalue = 0.0;

    Pvalue=0.0;
    for (int c = 0; c < Channel; ++c) {

        subTileM[tx] = mask[map*(K*K*Channel) + c*(K*K) + tx];
        subTileN[tx] = B[c* ImgBlockSize + tx*(ImgW) + imgCols ];
        __syncthreads();
        if(tx==0){
            for (int k = 0; k < maskSize; ++k)
                    Pvalue += subTileM[k] * subTileN[k];
        }
        __syncthreads();
    }
     C[map*ImgW + imgCols] = Pvalue;
}
 */

/* for test purposes, wo shared memory
__global__ void matrixMultiply(float *mask, float *B, float *C,
                               int numARows, int numAColumns,
                               int numBRows, int numBColumns,
                               int numCRows, int numCColumns,  int Channel, int K, int H_out, int W_out) {
  //@@ Insert code to implement matrix multiplication here
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column idenx of d_P and d_N
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    if ((Row < numCRows) && (Col < numCColumns)) {
        float Pvalue = 0;
        for (int k = 0; k < numAColumns; ++k)
            // each thread computes one element of the block sub-matrix
            Pvalue += mask[Row*numAColumns+k] * B[k*numBColumns+Col];
        C[Row*numCColumns+Col] = Pvalue;
    }
}
*/

__global__ void conv_forward_kernel(float *X_unroll, const float *X, const float *none,
     const int Batch, const int Map_out, const int C, const int H, const int W, const int K)
{
    int c,s,h_out, w_out, p,q, w_unroll, w_base, h_unroll;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H - K + 1;
    int W_out = W - K + 1;
    int W_unroll = H_out * W_out;

    //#define X_unroll( i2, i1, i0) X_unroll[(i2) * (W_unroll*K*K) + (i1) * (W_unroll) + i0]
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

__global__ void matrixMultiply(float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{

    __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // num of tiles in A and B both directions
    int num_tiles = ceil((1.0*numBRows)/TILE_WIDTH);

    // Identify the row and column of the P element to work on
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0.0;
    // Loop over the M and N tiles required to compute the P element
    // The code assumes that the Width is a multiple of TILE_WIDTH!

    Pvalue=0.0;
    for (int q = 0; q < num_tiles; ++q) {
        // Collaborative loading of M and N tiles into shared memory
        int col = q* TILE_WIDTH + tx;
        int row = q * TILE_WIDTH + ty;
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

        //if(Row < numCRows && Col < numCColumns) C[Row*numCColumns+Col] = Pvalue;
    }
     if(Row < numCRows && Col < numCColumns) C[Row*numCColumns+Col] = Pvalue;
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

    int inputSize= W * H * C; // per batch
    int outputSize= W_out * H_out * Map_out; // per batch
    int maskSize = K * K * Map_out * C;

    float *device_X_unrolled;
    float *device_X, *device_output;
    //float *device_mask;
    //errCheck(cudaMalloc((void **) &device_mask, maskSize * sizeof(float)));
    errCheck(cudaMalloc((void **) &device_X_unrolled, W_unroll * H_unroll * sizeof(float)));
    errCheck(cudaMalloc((void **) &device_output, outputSize * sizeof(float)));

    // errCheck(cudaMemcpy(device_mask, host_mask, maskSize * sizeof(float), cudaMemcpyHostToDevice));
    errCheck(cudaMemcpyToSymbol(mask, host_mask, maskSize*sizeof(float)));

    int num_threads = C * H_out * W_out;
    int num_blocks = ceil((1.0 * num_threads) / MAX_NUM_THREADS);
    /*   for bad solution
        dim3 dimGrid(Map_out, W_unroll, 1);
        dim3 dimBlock(K*K, 1, 1);
          for non-tiled solution
        dim3 dimGrid(W_unroll, Map_out, 1);
        dim3 dimBlock(1, 1, 1);
        matrixMultiply<<<dimGrid, dimBlock>>>( device_mask,  device_X_unrolled, device_output , Map_out, H_unroll, H_unroll, W_unroll, Map_out, W_unroll, C,  K,  H_out,  W_out);
    */

    errCheck(cudaMalloc((void **) &device_X, inputSize * sizeof(float)));

    cudaStream_t  stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    //errCheck(cudaMemcpy(device_X, host_input, Batch*inputSize * sizeof(float), cudaMemcpyHostToDevice));

    for (int n=0; n < Batch; n++) {
        errCheck(cudaMemcpyAsync(device_X, &host_input[n*inputSize], inputSize*sizeof(float), cudaMemcpyHostToDevice, stream0));
        // unroll kernel actually, therefore will not be using map or batch count
        conv_forward_kernel<<<num_blocks, MAX_NUM_THREADS, 0 , stream0>>>(device_X_unrolled, device_X, NULL,Batch,Map_out,C,  H,  W,  K);

        dim3 gridDim (ceil(1.0 * W_unroll / TILE_WIDTH),  ceil(1.0 *  Map_out/ TILE_WIDTH), 1);
        dim3 blockDim (TILE_WIDTH, TILE_WIDTH, 1);
        //forward_kernel<<<gridDim, blockDim>>>(device_mask,  device_X_unrolled, device_output, H_unroll, Map_out, W_unroll);
        matrixMultiply<<<gridDim, blockDim, 0, stream0>>>( device_X_unrolled, device_output, Map_out,
                               H_unroll, H_unroll,
                               W_unroll, Map_out,
                               W_unroll) ;
        errCheck(cudaMemcpyAsync((void *) (&host_output[n*outputSize]), device_output, outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream0));
        //errCheck(cudaMemcpy((void *) (&host_output[n*outputSize]), device_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    }

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    // Free device memory
    errCheck(cudaFree(device_output));
    errCheck(cudaFree(device_X_unrolled));
    errCheck(cudaFree(device_X));
    //errCheck(cudaFree(device_mask));


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
