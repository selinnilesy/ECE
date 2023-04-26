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
#define TILE_WIDTH 8

__global__ void conv_forward_kernel(float *output, const float *input, const float *none,
     const int Batch, const int Map_out, const int Channel,
     const int Height, const int Width, const int K)
{
    __shared__ float tile[TILE_WIDTH+6][TILE_WIDTH+6];

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

    float Pvalue = 0.0f;
    for(int c=0; c<Channel;c++){
        if((iX >= -3) && (iX < Width_out+3) && (iY >= -3)  && (iY < Height_out+3) )
        tile[ty][tx] = in_4d(imgId, c, iY+3,iX+3);

        __syncthreads (); // wait for tile

        if(ty < TILE_WIDTH && tx <TILE_WIDTH){
            for(int j = 0; j < K; j++) {
                for(int k = 0; k < K; k++) {
                    Pvalue  += mask_4d(m, c, j,k) * tile[j+ty][k+tx];
                }
            }
            if((oY < Height_out) && (oX < Width_out))
                out_4d(imgId, m, oY, oX) = Pvalue;
        }
         __syncthreads (); // wait for tile
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

    dim3 blockDim(TILE_WIDTH+ K - 1, TILE_WIDTH+ K - 1 , 1);
    dim3 gridDim(Map_out, W_grid*H_grid, Batch);
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

/*
 *     // First, create a cuBLAS handle:
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        return EXIT_FAILURE;
    }

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

     cublasSetMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb);
    stat = cublasSetMatrix (M, N, sizeof(*a), a, M, devPtrA, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data download failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }

    float alpha==1.0;
    float beta==0.0;
    cublasStatus_t cublasSgemm( handle, CUBLAS_OPT_T, CUBLAS_OPT_T,
                           K, K, K,
                           &alpha,
                           const float           *A, &K,
                           const float           *B, &K,
                           &beta,
                           NULL, int ldc)

    stat = cublasGetMatrix (M, N, sizeof(*a), devPtrA, M, a, M);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("data upload failed");
        cudaFree (devPtrA);
        cublasDestroy(handle);
        return EXIT_FAILURE;
    }
    cudaFree (devPtrA);
    cublasDestroy(handle);
    for (j = 1; j <= N; j++) {
        for (i = 1; i <= M; i++) {
            printf ("%7.0f", a[IDX2F(i,j,M)]);
        }
        printf ("\n");
    }
 */