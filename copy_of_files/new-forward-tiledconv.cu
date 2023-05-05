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

//__constant__ float mask[3136];
#define TILE_WIDTH_16 16
//#define TILE_WIDTH_32 32
// do padding for the dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
#define MATRIX_M1 16 // 4 = MATRIX_M2
#define MATRIX_N1 640 // = orig, 80*80=640
#define MATRIX_K1 64 // 1*7*7 = 49
#define MATRIX_M2 16
#define MATRIX_N2 912 // 30*30 ?? CONFIRM
#define MATRIX_K2 208 // 4*7*7 = 196

__global__ void half_to_float(__half *in_array, float *out, const int Batch, const int Map_out, const int Channel,
     const int Height, const int Width, const int K)
{
     const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int imgId = blockIdx.z;
    int m = blockIdx.x;
    const int W_grid=ceil((1.0*Width_out)/TILE_WIDTH);

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int by = (blockIdx.y / W_grid);
    int bx = (blockIdx.y % W_grid);
     #define out_4d(i3, i2, i1, i0) out[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
     #define in_4d(i3, i2, i1, i0) in_array[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]

    int oX = bx * TILE_WIDTH + tx;
    int oY = by * TILE_WIDTH + ty;

    if(ty < TILE_WIDTH && tx <TILE_WIDTH & oY < Height_out && (oX < Width_out))
             out_4d(imgId, m, oY, oX) = __half2float(in_4d(imgId, m, oY, oX));
    #undef out_4d
    #undef in_4d
}
__global__ void conv_forward_kernel(__half *output, __half *input, __half *mask,
     const int Batch, const int Map_out, const int Channel,
     const int Height, const int Width, const int K)
{
    __shared__ __half tile[TILE_WIDTH+6][TILE_WIDTH+6];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int imgId = blockIdx.z;
    int m = blockIdx.x;
    const int W_grid=ceil((1.0*Width_out)/TILE_WIDTH);

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int by = (blockIdx.y / W_grid);
    int bx = (blockIdx.y % W_grid);

    int oX = bx * TILE_WIDTH + tx;
    int oY = by * TILE_WIDTH + ty;

    int iX = oX-3; // MASK_WIDTH
    int iY = oY-3; // radius=3

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    __half Pvalue = 0.0f;
    for(int c=0; c<Channel;c++){
        if((iX >= -3) && (iX < Width_out+3) && (iY >= -3)  && (iY < Height_out+3) )
            tile[ty][tx] = in_4d(imgId, c, iY+3,iX+3);

        __syncthreads (); // wait for tile

        if(ty < TILE_WIDTH && tx <TILE_WIDTH){
            Pvalue  = mask_4d(m, c, 0,0) * tile[0+ty][0+tx]
                  + mask_4d(m, c, 0,1) * tile[0+ty][1+tx]
                  + mask_4d(m, c, 0,2) * tile[0+ty][2+tx]
                  + mask_4d(m, c, 0,3) * tile[0+ty][3+tx]
                  + mask_4d(m, c, 0,4) * tile[0+ty][4+tx]
                  + mask_4d(m, c, 0,5) * tile[0+ty][5+tx]
                  + mask_4d(m, c, 0,6) * tile[0+ty][6+tx]
                  + mask_4d(m, c, 1,0) * tile[1+ty][0+tx]
                  + mask_4d(m, c, 1,1) * tile[1+ty][1+tx]
                  + mask_4d(m, c, 1,2) * tile[1+ty][2+tx]
                   + mask_4d(m, c,1,3) * tile[1+ty][3+tx]
                  + mask_4d(m, c, 1,4) * tile[1+ty][4+tx]
                  + mask_4d(m, c, 1,5) * tile[1+ty][5+tx]
                  + mask_4d(m, c, 1,6) * tile[1+ty][6+tx]
                  + mask_4d(m, c, 2,0) * tile[2+ty][0+tx]
                  + mask_4d(m, c, 2,1) * tile[2+ty][1+tx]
                   + mask_4d(m, c,2,2) * tile[2+ty][2+tx]
                  + mask_4d(m, c, 2,3) * tile[2+ty][3+tx]
                  + mask_4d(m, c, 2,4) * tile[2+ty][4+tx]
                  + mask_4d(m, c, 2,5) * tile[2+ty][5+tx]
                  + mask_4d(m, c, 2,6) * tile[2+ty][6+tx]
                  + mask_4d(m, c, 3,0) * tile[3+ty][0+tx]
                   + mask_4d(m, c,3,1) * tile[3+ty][1+tx]
                  + mask_4d(m, c, 3,2) * tile[3+ty][2+tx]
                  + mask_4d(m, c, 3,3) * tile[3+ty][3+tx]
                  + mask_4d(m, c, 3,4) * tile[3+ty][4+tx]
                  + mask_4d(m, c, 3,5) * tile[3+ty][5+tx]
                  + mask_4d(m, c, 3,6) * tile[3+ty][6+tx]
                   + mask_4d(m, c, 4,0) * tile[4+ty][0+tx]
                   + mask_4d(m, c,4,1) * tile[4+ty][1+tx]
                  + mask_4d(m, c, 4,2) * tile[4+ty][2+tx]
                  + mask_4d(m, c, 4,3) * tile[4+ty][3+tx]
                  + mask_4d(m, c, 4,4) * tile[4+ty][4+tx]
                  + mask_4d(m, c, 4,5) * tile[4+ty][5+tx]
                  + mask_4d(m, c, 4,6) * tile[4+ty][6+tx]
                   + mask_4d(m, c, 5,0) * tile[5+ty][0+tx]
                   + mask_4d(m, c,5,1) * tile[5+ty][1+tx]
                  + mask_4d(m, c, 5,2) * tile[5+ty][2+tx]
                  + mask_4d(m, c, 5,3) * tile[5+ty][3+tx]
                  + mask_4d(m, c, 5,4) * tile[5+ty][4+tx]
                  + mask_4d(m, c, 5,5) * tile[5+ty][5+tx]
                  + mask_4d(m, c, 5,6) * tile[5+ty][6+tx]
                   + mask_4d(m, c, 6,0) * tile[6+ty][0+tx]
                   + mask_4d(m, c,6,1) * tile[6+ty][1+tx]
                  + mask_4d(m, c, 6,2) * tile[6+ty][2+tx]
                  + mask_4d(m, c, 6,3) * tile[6+ty][3+tx]
                  + mask_4d(m, c, 6,4) * tile[6+ty][4+tx]
                  + mask_4d(m, c, 6,5) * tile[6+ty][5+tx]
                  + mask_4d(m, c, 6,6) * tile[6+ty][6+tx];
        }
        __syncthreads (); // wait for tile
    }
    if(ty < TILE_WIDTH && tx <TILE_WIDTH & oY < Height_out && (oX < Width_out))
            out_4d(imgId, m, oY, oX) = Pvalue;

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
     const int W_grid=ceil((1.0* Height_out)/TILE_WIDTH);
    const int H_grid=ceil((1.0* Width_out)/TILE_WIDTH);

    int inputSize = Batch * Channel * Height * Width;
    int maskSize = Map_out * Channel * K * K;
     int outputSize = Batch * Map_out * Height_out * Width_out;

     std::cout<<"Height_out "<< Height_out <<std::endl;
    std::cout<<"Width_out "<< Width_out <<std::endl;
    std::cout<<"Channel "<< Channel <<std::endl;
    std::cout<<"K "<< K <<std::endl;
    std::cout<<"Map_out "<< Map_out <<std::endl;

    __half *h_host_input, *h_host_mask, *half_device_input_ptr, *half_device_output_ptr, *half_device_mask;

    h_host_mask = (__half*) malloc(MATRIX_M1*MATRIX_K1*sizeof(__half));
    if(Channel * K * K== 640){
        for (int i=0; i<MATRIX_M1; i++){
            for (int j=0; j<MATRIX_K1; j++)
                if(i<Map_out &&) h_host_mask[i] = __float2half(host_mask[i]);
                else h_host_mask[i] = __float2half(0.0);
        }
        h_host_input = (__half*) malloc(MATRIX_K1*MATRIX_N1*sizeof(__half));
        for (int i=0; i<MATRIX_K1*MATRIX_N1; i++)
            if(i<Map_out) h_host_input[i] = __float2half(host_input[i]);
            else h_host_input[i] = __float2half(0.0);
        }
    else{
        for (int i=0; i<MATRIX_M1*MATRIX_K1; i++)
            if(i<Map_out) h_host_mask[i] = __float2half(host_mask[i]);
            else h_host_mask[i] = __float2half(0.0);

        h_host_input = (__half*) malloc(MATRIX_K1*MATRIX_N1*sizeof(__half));
        for (int i=0; i<MATRIX_K1*MATRIX_N1; i++)
            if(i<Map_out) h_host_input[i] = __float2half(host_input[i]);
            else h_host_input[i] = __float2half(0.0);
    }
    }

    errCheck(cudaMalloc((void **) &half_device_input_ptr, inputSize * sizeof(__half)));
    errCheck(cudaMalloc((void **) &half_device_mask, maskSize * sizeof(__half)));
    errCheck(cudaMalloc((void **) &half_device_output_ptr, outputSize * sizeof(__half)));
    errCheck(cudaMalloc((void **) device_output_ptr, outputSize * sizeof(float)));


    errCheck(cudaMemcpy(half_device_input_ptr, h_host_input, inputSize * sizeof(__half), cudaMemcpyHostToDevice));
    errCheck(cudaMemcpy(half_device_mask, h_host_mask, maskSize * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_WIDTH+ K - 1, TILE_WIDTH+ K - 1 , 1);
    dim3 gridDim(Map_out, W_grid*H_grid, Batch);
    conv_forward_kernel<<< gridDim, blockDim >>>(half_device_output_ptr,half_device_input_ptr, half_device_mask, Batch, Map_out, Channel, Height,Width, K );

    cudaDeviceSynchronize();

    half_to_float<<<gridDim, blockDim>>>(half_device_output_ptr, *device_output_ptr,  Batch, Map_out, Channel, Height,Width, K );
    errCheck(cudaMemcpy((void*) host_output, *device_output_ptr, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

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
    /*
    std::cout<<" Height_out: "<< Height_out <<std::endl;
    std::cout<<" Width_out: "<< Width_out <<std::endl;
    std::cout<<" Height: "<< Height <<std::endl;
    std::cout<<" Width: "<< Width <<std::endl;
     */

    // Free device memory
    //errCheck(cudaFree(device_output));
    //errCheck(cudaFree(device_input));
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
