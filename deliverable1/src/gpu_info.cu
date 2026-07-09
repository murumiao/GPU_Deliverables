#include <stdio.h>
#include <cuda_runtime.h>

int main() {

    int deviceCount = 0;

    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {

        cudaDeviceProp prop;

        err = cudaGetDeviceProperties(&prop, i);

        if (err != cudaSuccess) {
            printf("Failed getting properties for device %d: %s\n",
                   i,
                   cudaGetErrorString(err));
            continue;
        }

        printf("=====================================\n");
        printf("GPU %d : %s\n", i, prop.name);
        printf("=====================================\n");

        printf("Compute Capability: %d.%d\n",
               prop.major,
               prop.minor);

        printf("Global Memory: %.2f GB\n",
               (float)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

        printf("Shared Memory per Block: %zu KB\n",
               prop.sharedMemPerBlock / 1024);

        printf("Registers per Block: %d\n",
               prop.regsPerBlock);

        printf("Warp Size: %d\n",
               prop.warpSize);

        printf("Max Threads per Block: %d\n",
               prop.maxThreadsPerBlock);

        printf("Max Thread Dimensions: %d x %d x %d\n",
               prop.maxThreadsDim[0],
               prop.maxThreadsDim[1],
               prop.maxThreadsDim[2]);

        printf("Max Grid Size: %d x %d x %d\n",
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);

        printf("Multiprocessors: %d\n",
               prop.multiProcessorCount);

        printf("Clock Rate: %.2f MHz\n",
               prop.clockRate / 1000.0);

        printf("Memory Clock Rate: %.2f MHz\n",
               prop.memoryClockRate / 1000.0);

        printf("Memory Bus Width: %d bits\n",
               prop.memoryBusWidth);

        printf("L2 Cache Size: %d KB\n",
               prop.l2CacheSize / 1024);

        printf("Concurrent Kernels: %d\n",
               prop.concurrentKernels);

        printf("Unified Addressing: %d\n",
               prop.unifiedAddressing);

        printf("\n");
    }

    return 0;
}