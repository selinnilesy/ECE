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

__global__ void conv_forward_kernel(float *output, const float *input, const float *none,
     const int Batch, const int Map_out, const int Channel,
     const int Height, const int Width, const int K)
{
    __shared__ float tileMatA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileMatB[TILE_WIDTH][TILE_WIDTH];

    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int column = blockIdx.x * TILE_WIDTH + tx;

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    int numMatAColumns = Channel*K*K;
    float acc=0.0;

    int numIterations = ceil((numMatAColumns*1.0)/TILE_WIDTH);

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    for(int i=0; i<numIterations; i++){
        int tempCol = i*TILE_WIDTH+tx;
        int tempRow = i*TILE_WIDTH+ty;
        tileMatA[ty][tx] = 0.0;
        tileMatB[ty][tx] = 0.0;

        int W_m = row;
        int W_c = tempCol/(K*K);
        int W_h = (tempCol%(K*K))/K;
        int W_w = (tempCol%(K*K))%K;

        if((tempCol < numMatAColumns) && (row < Map_out) )
            tileMatA[ty][tx] = mask_4d(W_m, W_c, W_h, W_w);
        else tileMatA[ty][tx] = 0.0;

        int X_b = b;
        int X_c = tempRow/(K*K);
        int X_p = (tempRow%(K*K))/K;
        int X_q = (tempRow%(K*K))%K;
        int X_h = column / Width_out;
        int X_w = column % Width_out;

        if((tempRow < numMatAColumns) && (column < Height_out*Width_out) )
            tileMatB[ty][tx] = in_4d(X_b, X_c, X_p + X_h, X_q + X_w);
        else tileMatB[ty][tx] = 0.0;

        __syncthreads();

        for(int q = 0; q < TILE_WIDTH; q++) {
            acc  += tileMatA[ty][q] * tileMatB[q][tx];
        }

        __syncthreads ();
    }

    int Y_b = b;
    int Y_m = row;
    int Y_h = column / Width_out;
    int Y_w = column % Width_out;
    if((row < Map_out) && (column < Height_out*Width_out))
            out_4d(Y_b, Y_m, Y_h, Y_w) = acc;

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int inputSize = Batch * Channel * Height * Width;
    int maskSize = Map_out * Channel * K * K;
     int outputSize = Batch * Map_out * Height_out * Width_out;


    errCheck(cudaMalloc((void **) device_input_ptr, inputSize * sizeof(float)));
    errCheck(cudaMalloc((void **) device_output_ptr, outputSize * sizeof(float)));

    errCheck(cudaMemcpy(*device_input_ptr, host_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    errCheck(cudaMemcpyToSymbol(mask, host_mask, maskSize*sizeof(float)));


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

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int W_grid=ceil((1.0* Height_out)/TILE_WIDTH);
    const int H_grid=ceil((1.0* Width_out)/TILE_WIDTH);
    //std::cout<<"W_grid: "<<W_grid<<std::endl;
    //std::cout<<"H_grid: "<<H_grid<<std::endl;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH , 1);
    dim3 gridDim(ceil((1.0* Width_out* Height_out)/TILE_WIDTH), ceil((1.0*Map_out)/TILE_WIDTH), Batch);
    conv_forward_kernel<<< gridDim, blockDim >>>(device_output,device_input, NULL, Batch, Map_out, Channel, Height,Width, K );

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
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int outputSize = Batch * Map_out * Height_out * Width_out;
    errCheck(cudaMemcpy(host_output, device_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    errCheck(cudaFree(device_output));
    errCheck(cudaFree(device_input));
    //errCheck(cudaFree(device_mask));

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
