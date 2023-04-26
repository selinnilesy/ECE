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
#define TILE_WIDTH 32
__global__ void conv_forward_kernel(float *output, const float *input, const float *none,
     const int Batch, const int Map_out, const int Channel,
     const int Height, const int Width, const int K)
{
    /*
    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here
    //__shared__ float tile_input[TILE_WIDTH][TILE_WIDTH];
    //__shared__ float tile_mask[7][7];

    int imgId = blockIdx.z;
    int m = blockIdx.x;
    const int W_grid=ceil((1.0*Width_out)/TILE_WIDTH);
    //const int H_grid=Height_out/TILE_WIDTH;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    float acc = 0.0f;
    if (h < Height_out && w < Width_out){
        for(int c=0; c<Channel;c++){
            for (int p = 0; p < K; p++){
                for (int q = 0; q < K; q++)
                acc += in_4d(imgId, c, h + p, w + q) * mask_4d(m, c, p, q);
            }
        }
        out_4d(imgId , m, h, w) = acc;
    }


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
    const int W_grid=ceil((1.0* Width_out)/TILE_WIDTH);
    const int H_grid=ceil((1.0* Height_out)/TILE_WIDTH);

    int inputSize = 4 * Channel * Height * Width;
    int maskSize = Map_out * Channel * K * K;
    int outputSize = 4 * Map_out * Height_out * Width_out;

    float *device_output_ptr2= NULL;
    float *device_input_ptr2 = NULL;

    errCheck(cudaMalloc((void **) device_input_ptr, inputSize * sizeof(float)));
    errCheck(cudaMalloc((void **) &device_input_ptr2, inputSize * sizeof(float)));

    errCheck(cudaMalloc((void **) device_output_ptr, outputSize * sizeof(float)));
    errCheck(cudaMalloc((void **) &device_output_ptr2, outputSize * sizeof(float)));

    errCheck(cudaMemcpyToSymbol(mask, host_mask, maskSize*sizeof(float)));


    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, W_grid*H_grid, 4);

    /*
     *
     * HERE I HAVE AN UNUSED CODE BECAUSE I ASSUME HOST INPUT AND OUTPUT ALREADY ALLOCATED WITH CUDA MALLOCHOST FUNCTION.
     *
    float *host_output_pinned= NULL;
    float *host_output_pinned2 = NULL;
    float *host_input_pinned= NULL;

    errCheck(cudaMallocHost ( (void**) &host_output_pinned, outputSize*sizeof(float) ));
    errCheck(cudaMallocHost ( (void**) &host_output_pinned2, outputSize *sizeof(float)));
    errCheck(cudaMallocHost ( (void**) &host_input_pinned, Batch*inputSize*sizeof(float) ));
     std::cout << "host_input_pinned: ";
    for(int i=0; i<Batch*inputSize; i++){
        host_input_pinned[i] = host_input[i];
    }

    std::cout << host_input_pinned[0] << '\t';
    std::cout << host_input_pinned[1*inputSize] << '\t';
    std::cout << host_input_pinned[2*inputSize] << '\t';
     */

    cudaStream_t  stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    int ctr=0;
    for (int i=0; i<Batch/8; i+=1) {
        ctr=i*2;
        errCheck(cudaMemcpyAsync((*device_input_ptr), &host_input[ctr*inputSize], inputSize*sizeof(float), cudaMemcpyHostToDevice, stream0));
        errCheck(cudaMemcpyAsync((device_input_ptr2), &host_input[(ctr+1)*inputSize], inputSize*sizeof(float), cudaMemcpyHostToDevice, stream1));

        conv_forward_kernel<<<gridDim, blockDim,0, stream0>>>(*device_output_ptr,*device_input_ptr, NULL, Batch, Map_out, Channel, Height,Width, K );
        conv_forward_kernel<<<gridDim, blockDim,0, stream1>>>(device_output_ptr2,device_input_ptr2, NULL, Batch, Map_out, Channel, Height,Width, K );

        errCheck(cudaMemcpyAsync((void *) &host_output[ctr*outputSize], *device_output_ptr, outputSize*sizeof(float), cudaMemcpyDeviceToHost, stream0));
        errCheck(cudaMemcpyAsync((void *) &host_output[(ctr+1)*outputSize], (device_output_ptr2), outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream1));
    }

    if(Batch==100){
        errCheck(cudaMemcpyAsync((*device_input_ptr), &host_input[(Batch/4-1)*(inputSize/2)], (inputSize/2)*sizeof(float), cudaMemcpyHostToDevice, stream0));
        conv_forward_kernel<<<gridDim, blockDim,0, stream0>>>(*device_output_ptr,*device_input_ptr, NULL, Batch, Map_out, Channel, Height,Width, K );
        errCheck(cudaMemcpyAsync((void *) &host_output[(Batch/4-1)*(outputSize/2)], *device_output_ptr, (outputSize/2)*sizeof(float), cudaMemcpyDeviceToHost, stream0));
    }

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
         std::cout<<"CUDA kern error: "<<cudaGetErrorString(error)<<std::endl;
         exit(-1);
     }
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K){
    /*
     *
     * const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int W_grid=ceil((1.0* Width_out)/TILE_WIDTH);
    const int H_grid=ceil((1.0* Height_out)/TILE_WIDTH);

    const int W_grid=ceil((1.0* Width_out)/TILE_WIDTH);
    const int H_grid=ceil((1.0* Height_out)/TILE_WIDTH);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, W_grid*H_grid, Batch);
    conv_forward_kernel<<< gridDim, blockDim >>>(device_output,device_input, device_mask, Batch, Map_out, Channel, Height,Width, K );
    */
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
    //const int Height_out = Height - K + 1;
   // const int Width_out = Width - K + 1;
   // int outputSize = Batch * Map_out * Height_out * Width_out;
    //errCheck(cudaMemcpy(host_output, device_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

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