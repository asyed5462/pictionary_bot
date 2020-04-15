#include "convolution.h"

// Roshini's tiled matrix multiply code from MP3
// TODO: remove wb utility functions + update any c --> c++ code
// #include <wb.h>

// #define wbCheck(stmt)                                                      \
//     do                                                                     \
//     {                                                                      \
//         cudaError_t err = stmt;                                            \
//         if (err != cudaSuccess)                                            \
//         {                                                                  \
//             wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
//             wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
//             return -1;                                                     \
//         }                                                                  \
//     } while (0)

// // Compute C = A * B
// __global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
//                                int numAColumns, int numBRows,
//                                int numBColumns, int numCRows,
//                                int numCColumns)
// {
//     //@@ Insert code to implement matrix multiplication here

//     // shared tiles for the two input matrices
//     extern __shared__ float tiles[];
//     float* aTile = tiles;
//     float* bTile = &(tiles[blockDim.x * blockDim.y]);
  
//     // local variables that will be used later -- saved as per-thread registers
//     int phase, innerProductIndex, aRow, aColumn, bRow, bColumn;
//     float innerProduct = 0; // inner product goes across phases, which is why it is initialized here

//     // the row and column in the output matrix, that this thread is mapped to
//     int cRow = blockIdx.y * blockDim.y + threadIdx.y;
//     int cColumn = blockIdx.x * blockDim.x + threadIdx.x;

//     // the index within the tile, that this thread is mapped to
//     int tileIndex = threadIdx.y * blockDim.x + threadIdx.x;

//     // collaboratively load the tiles for A and B
//     // meaning each thread loads one element in the aTile and one element in the bTile
//     for (phase = 0; (phase < numAColumns / blockDim.x) && (phase < numBRows / blockDim.y); phase++)
//     {
//         aRow = cRow;
//         aColumn = (phase * blockDim.x) + threadIdx.x;

//         // recall, aTile will contain the current phase of a set of rows from A
//         if ((aRow < numARows) && (aColumn < numAColumns))
//         {
//             aTile[tileIndex] = A[aRow * numAColumns + aColumn];
//         }
//         else
//         { // load 0 so that no effect on inner product from out of bounds values
//             aTile[tileIndex] = 0;
//         }

//         bRow = (phase * blockDim.y) + threadIdx.y;
//         bColumn = cColumn;

//         // recall, bTile will contain the current phase of a set of columns from B
//         if ((bRow < numBRows) && (bColumn < numBColumns))
//         {
//             bTile[tileIndex] = B[bRow * numBColumns + bColumn];
//         }
//         else
//         { // load 0 so that no effect on inner product from out of bounds values
//             bTile[tileIndex] = 0;
//         }

//         // sync threads -- tile is fully loaded!
//         __syncthreads();

//         // accumulate the inner product
//         for (innerProductIndex = 0; (innerProductIndex < blockDim.x) && (innerProductIndex < blockDim.y); innerProductIndex++)
//         { // walk along a row of the aTile and a column of the bTile
//             innerProduct += aTile[threadIdx.y * blockDim.x + innerProductIndex] * bTile[innerProductIndex * blockDim.x + threadIdx.x];
//         }

//         // sync threads -- all threads are done using the tile!
//         __syncthreads();
//     }

//     // save inner product to the output matrix if in bounds
//     if ((cRow < numCRows) && (cColumn < numCColumns))
//     {
//         C[cRow * numCColumns + cColumn] = innerProduct;
//     }

//     return;
// }

// int main(int argc, char **argv)
// {
//     wbArg_t args;
//     float *hostA; // The A matrix
//     float *hostB; // The B matrix
//     float *hostC; // The output C matrix
//     float *deviceA;
//     float *deviceB;
//     float *deviceC;
//     int numARows;    // number of rows in the matrix A
//     int numAColumns; // number of columns in the matrix A
//     int numBRows;    // number of rows in the matrix B
//     int numBColumns; // number of columns in the matrix B
//     int numCRows;    // number of rows in the matrix C (you have to set this)
//     int numCColumns; // number of columns in the matrix C (you have to set
//                      // this)

//     int TILE_WIDTH = 2;

//     args = wbArg_read(argc, argv);

//     wbTime_start(Generic, "Importing data and creating memory on host");
//     hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
//                               &numAColumns);
//     hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
//                               &numBColumns);
//     //@@ Set numCRows and numCColumns
//     numCRows = numARows;
//     numCColumns = numBColumns;

//     //@@ Allocate the hostC matrix
//     hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));
//     wbTime_stop(Generic, "Importing data and creating memory on host");

//     wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
//     wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

//     wbTime_start(GPU, "Allocating GPU memory.");
//     //@@ Allocate GPU memory here
//     cudaMalloc((void **)(&deviceA), numARows * numAColumns * sizeof(float));
//     cudaMalloc((void **)(&deviceB), numBRows * numBColumns * sizeof(float));
//     cudaMalloc((void **)(&deviceC), numCRows * numCColumns * sizeof(float));

//     wbTime_stop(GPU, "Allocating GPU memory.");

//     wbTime_start(GPU, "Copying input memory to the GPU.");
//     //@@ Copy memory to the GPU here
//     cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(deviceC, hostC, numCRows * numCColumns * sizeof(float), cudaMemcpyHostToDevice);

//     wbTime_stop(GPU, "Copying input memory to the GPU.");

//     //@@ Initialize the grid and block dimensions here
//     dim3 gridDim(ceil((float)numCColumns / (float)TILE_WIDTH), ceil((float)numCRows / (float)TILE_WIDTH), 1);
//     dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

//     wbTime_start(Compute, "Performing CUDA computation");
//     //@@ Launch the GPU Kernel here
//     matrixMultiply<<<gridDim, blockDim, TILE_WIDTH * TILE_WIDTH * 2 * sizeof(float)>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
//     wbCheck(cudaGetLastError());

//     cudaDeviceSynchronize();
//     wbTime_stop(Compute, "Performing CUDA computation");

//     wbTime_start(Copy, "Copying output memory to the CPU");
//     //@@ Copy the GPU memory back to the CPU here
//     cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost);

//     wbTime_stop(Copy, "Copying output memory to the CPU");

//     wbTime_start(GPU, "Freeing GPU Memory");
//     //@@ Free the GPU memory here
//     cudaFree(deviceA);
//     cudaFree(deviceB);
//     cudaFree(deviceC);

//     wbTime_stop(GPU, "Freeing GPU Memory");

//     wbSolution(args, hostC, numCRows, numCColumns);

//     free(hostA);
//     free(hostB);
//     free(hostC);

//     return 0;
// }
