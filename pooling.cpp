#include "pooling.h"
#include <cuda.h>

#define NEIGHBORHOOD_WIDTH 2

dim3 gridDim(numberOfInputImages);
dim3 blockDim(196);

// ReLU is a nonlinear activation function
__device__ float ReLU(float value)
{
    if (value < 0)
    {
        return 0;
    }
    return value;
}

// MAXPOOLING + RELU
// Purpose:
// Reduces each input image to an output image that contains the 
//  maximum value from each NEIGHBORHOOD_WIDTH * NEIGHBORHOOD_WIDTH
//  region. Also applies ReLU.

// Assumptions:
// Input images MUST be square with even dimensions!

// Input Specs:
// inputImages/outputImages = images stored consecutively in memory,
//  each image is stored in row-major order.
__global__ void maxPooling(float *inputImages, float *outputImages, int inputImageWidth)
{
    // each thread is responsible for one 2x2 tile of the input
    int outputImageWidth = inputImageWidth / NEIGHBORHOOD_WIDTH;

    // image index with respect to the start of 'inputImages'
    int imageIndex = blockIdx.z * blockDim.z + threadIdx.z;

    // row and column with respect to the input image
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // read four values in the local neighborhood
    float max = 0.0;
    for (int x = 0; x < NEIGHBORHOOD_WIDTH; x++)
    {
        for (int y = 0; y < NEIGHBORHOOD_WIDTH; y++)
        {
            // calculate index of the pixel with respect to the start of 'inputImages'
            int index = (imageIndex * inputImageWidth * inputImageWidth) + ((row * 2 + y) * inputImageWidth) + (col * 2 + x);

            // update max if necessary
            if (inputImages[index] > max)
            {
                max = inputImages[index]
            }
        }
    }

    // write the maximum to outputImages
    int outputIndex = (imageIndex * outputImageWidth * outputImageWidth) + (row * outputImageWidth) + col;

    outputImages[outputIndex] = ReLU(max);
}