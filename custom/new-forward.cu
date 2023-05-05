#include <iostream>
#include <stdint.h>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"
#include <mma.h>
using namespace nvcuda;

// do padding for the dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
#define MATRIX_M 16
#define MATRIX_N 912
#define MATRIX_K 208

#define errCheck(ans) { checkError((ans), __FILE__, __LINE__); }
inline void checkError(cudaError_t err, const char * file, int line, bool abort = true) {
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU Error: %s --> %s:%d\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
}

#define TILE_WIDTH_16 16
//#define TILE_WIDTH_32 32
__global__ void conv_forward_kernel_16(float *output, __half *input, __half *mask,
     const int Batch, const int Map_out, const int Channel,
     const int Height, const int Width, const int K)
{
    //__shared__ __half tileA[TILE_WIDTH_16][TILE_WIDTH_16];
    //__shared__ __half tileB[TILE_WIDTH_16][TILE_WIDTH_16];

    float alpha = 1.0;
    float beta=0.0;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH_16 + ty;
    int column = blockIdx.x * TILE_WIDTH_16 + tx;

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    int numMatAColumns = Channel*K*K;
    __half acc=0.0;

    int numIterations = ceil((numMatAColumns*1.0)/TILE_WIDTH_16);

    #define out_4d(i3, i2, i1, i0) &output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) &input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) &mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int lda = Map_out;
    int ldb = Channel*K*K;
    int ldc = Map_out;

    for(int i=0; i<numIterations; i++){
        int tempCol = i*TILE_WIDTH_16+tx;
        int tempRow = i*TILE_WIDTH_16+ty;

        int W_m = row;
        int W_c = tempCol/(K*K);
        int W_h = (tempCol%(K*K))/K;
        int W_w = (tempCol%(K*K))%K;

        if((tempCol < numMatAColumns) && (row < Map_out) )
            wmma::load_matrix_sync(a_frag, mask_4d(W_m, W_c, W_h, W_w), lda);

        int X_b = b;
        int X_c = tempRow/(K*K);
        int X_p = (tempRow%(K*K))/K;
        int X_q = (tempRow%(K*K))%K;
        int X_h = column / Width_out;
        int X_w = column % Width_out;

        if((tempRow < numMatAColumns) && (column < Height_out*Width_out) )
            wmma::load_matrix_sync(b_frag, in_4d(X_b, X_c, X_p + X_h, X_q + X_w), ldb);

        // Perform the matrix multiplication
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    int Y_b = b;
    int Y_m = row;
    int Y_h = column / Width_out;
    int Y_w = column % Width_out;

    if ((row < Map_out) && (column < Height_out*Width_out)) {
        wmma::load_matrix_sync(c_frag, out_4d(Y_b, Y_m, Y_h, Y_w), ldc, wmma::mem_row_major);

        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(out_4d(Y_b, Y_m, Y_h, Y_w), c_frag, ldc, wmma::mem_row_major);
   }

    #undef out_4d
    #undef in_4d
    #undef mask_4d
}
/*
 __global__ void half_to_float(__half *in_array, float *out,  int outlen, int M)
{
    const int map = threadIdx.y + blockDim.y*blockIdx.y;
    const int z = blockIdx.z;
    const int houtwout = threadIdx.x + blockDim.x*blockIdx.x;
    if(map < M && houtwout < outlen) out[z*outlen*M + map*outlen+houtwout] = __half2float(in_array[z*outlen*M + map*outlen+houtwout]);
}

__global__ void conv_forward_kernel_32(__half *output, __half *input, __half *mask,
     const int Batch, const int Map_out, const int Channel,
     const int Height, const int Width, const int K)
{
    __shared__ __half tileA[TILE_WIDTH_32][TILE_WIDTH_32];
    __shared__ __half tileB[TILE_WIDTH_32][TILE_WIDTH_32];

    int b = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH_32 + ty;
    int column = blockIdx.x * TILE_WIDTH_32 + tx;

    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    int numMatAColumns = Channel*K*K;
    __half acc=0.0;

    int numIterations = ceil((numMatAColumns*1.0)/TILE_WIDTH_32);

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    for(int i=0; i<numIterations; i++){
        int tempCol = i*TILE_WIDTH_32+tx;
        int tempRow = i*TILE_WIDTH_32+ty;
        tileA[ty][tx] = 0.0;
        tileB[ty][tx] = 0.0;

        int W_m = row;
        int W_c = tempCol/(K*K);
        int W_h = (tempCol%(K*K))/K;
        int W_w = (tempCol%(K*K))%K;

        if((tempCol < numMatAColumns) && (row < Map_out) )
            tileA[ty][tx] = mask_4d(W_m, W_c, W_h, W_w);
        else tileA[ty][tx] = 0.0;

        int X_b = b;
        int X_c = tempRow/(K*K);
        int X_p = (tempRow%(K*K))/K;
        int X_q = (tempRow%(K*K))%K;
        int X_h = column / Width_out;
        int X_w = column % Width_out;

        if((tempRow < numMatAColumns) && (column < Height_out*Width_out) )
            tileB[ty][tx] = in_4d(X_b, X_c, X_p + X_h, X_q + X_w);
        else tileB[ty][tx] = 0.0;

        __syncthreads();

        acc  += tileA[ty][0] * tileB[0][tx]
        +tileA[ty][1] * tileB[1][tx]
        +tileA[ty][2] * tileB[2][tx]
        +tileA[ty][3] * tileB[3][tx]
        +tileA[ty][4] * tileB[4][tx]
        +tileA[ty][5] * tileB[5][tx]
        +tileA[ty][6] * tileB[6][tx]
        +tileA[ty][7] * tileB[7][tx]
        +tileA[ty][8] * tileB[8][tx]
        +tileA[ty][9] * tileB[9][tx]
        +tileA[ty][10] * tileB[10][tx]
        +tileA[ty][11] * tileB[11][tx]
        +tileA[ty][12] * tileB[12][tx]
        +tileA[ty][13] * tileB[13][tx]
        +tileA[ty][14] * tileB[14][tx]
        +tileA[ty][15] * tileB[15][tx]
        +tileA[ty][16] * tileB[16][tx]
        +tileA[ty][17] * tileB[17][tx]
        +tileA[ty][18] * tileB[18][tx]
        +tileA[ty][19] * tileB[19][tx]
        +tileA[ty][20] * tileB[20][tx]
        +tileA[ty][21] * tileB[21][tx]
        +tileA[ty][22] * tileB[22][tx]
        +tileA[ty][23] * tileB[23][tx]
        +tileA[ty][24] * tileB[24][tx]
        +tileA[ty][25] * tileB[25][tx]
        +tileA[ty][26] * tileB[26][tx]
        +tileA[ty][27] * tileB[27][tx]
        +tileA[ty][28] * tileB[28][tx]
        +tileA[ty][29] * tileB[29][tx]
        +tileA[ty][30] * tileB[30][tx]
        +tileA[ty][31] * tileB[31][tx];

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
 */

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

    std::cout<<"Height_out "<< Height_out <<std::endl;
    std::cout<<"Width_out "<< Width_out <<std::endl;
    std::cout<<"Channel "<< Channel <<std::endl;
    std::cout<<"K "<< K <<std::endl;
    std::cout<<"Map_out "<< Map_out <<std::endl;

     __half *h_host_input, *h_host_mask, *half_device_input_ptr, *half_device_mask;

     h_host_input = (__half*) malloc(inputSize*sizeof(__half));
     for (int i=0; i<inputSize; i++)
        h_host_input[i] = __float2half(host_input[i]);

     h_host_mask = (__half*) malloc(maskSize*sizeof(__half));
     for (int i=0; i<maskSize; i++)
        h_host_mask[i] = __float2half(host_mask[i]);

    errCheck(cudaMalloc((void **) &half_device_input_ptr, inputSize * sizeof(__half)));
     errCheck(cudaMalloc((void **) &half_device_mask, maskSize * sizeof(__half)));
    errCheck(cudaMalloc((void **) device_output_ptr, outputSize * sizeof(float)));

    errCheck(cudaMemcpy(half_device_input_ptr, h_host_input, inputSize * sizeof(__half), cudaMemcpyHostToDevice));
    errCheck(cudaMemcpy(half_device_mask, h_host_mask, maskSize * sizeof(__half), cudaMemcpyHostToDevice));


    // First: using WMMA
   dim3 gridDim;
   dim3 blockDim;

   // blockDim.x must be a multple of warpSize
   // 128x4 means we have 16 warps and a block computes a 64x64 output tile
   blockDim.x = 128;
   blockDim.y = 4;

   gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
   gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

   cudaEvent_t startWMMA;
   cudaEvent_t stopWMMA;
   errCheck(cudaEventCreate(&startWMMA));
   errCheck(cudaEventCreate(&stopWMMA));
   /*
   cublasHandle_t cublasHandle;
   cublasErrCheck(cublasCreate(&cublasHandle));
   cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
    */

   printf("Running with wmma...\n");
   errCheck(cudaEventRecord(startWMMA));
   conv_forward_kernel_16 <<< gridDim, blockDim >>> (*device_output_ptr, h_host_input, h_host_mask, Batch, Map_out, Channel, Height,Width, K);
   errCheck(cudaEventRecord(stopWMMA));

   errCheck(cudaEventDestroy(startWMMA));
   errCheck(cudaEventDestroy(stopWMMA));


    //dim3 blockDim(TILE_WIDTH_16, TILE_WIDTH_16 , 1);
    //dim3 gridDim(ceil((1.0* Width_out* Height_out)/TILE_WIDTH_16), ceil((1.0*Map_out)/TILE_WIDTH_16), Batch);
    //conv_forward_kernel_16<<< gridDim, blockDim >>>(half_device_output_ptr, half_device_input_ptr, half_device_mask, Batch, Map_out, Channel, Height,Width, K );
    /*if(Map_out / 16 ){}
    else{
        dim3 blockDim(TILE_WIDTH_32, TILE_WIDTH_32 , 1);
        dim3 gridDim(ceil((1.0* Width_out* Height_out)/TILE_WIDTH_32), ceil((1.0*Map_out)/TILE_WIDTH_32), Batch);
        conv_forward_kernel_32<<< gridDim, blockDim >>>(half_device_output_ptr, half_device_input_ptr, half_device_mask, Batch, Map_out, Channel, Height,Width, K );
    }


    cudaDeviceSynchronize();

    dim3 b(TILE_WIDTH_32, TILE_WIDTH_32 , 1);
    dim3 g(ceil((1.0* Width_out* Height_out)/TILE_WIDTH_32), ceil((1.0*Map_out)/TILE_WIDTH_32), Batch);
    half_to_float<<<g, b>>>(half_device_output_ptr, *device_output_ptr, Height_out*Width_out, Map_out );
    errCheck(cudaMemcpy((void*) host_output, *device_output_ptr, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
     */


    //errCheck(cudaFree(half_device_output_ptr));
    //errCheck(cudaFree(half_device_input_ptr));
    //errCheck(cudaFree(device_output_ptr));
    //errCheck(cudaFree(half_device_mask));

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