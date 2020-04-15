#include <cuda.h>
#include </include/curand_kernel.h>

// RANDOMLY GENERATE BIASES FOR POOLING
// pooling round 1 - 28x28 images are reduced to 14x14 images = 196 biases per image
// pooling round 2 - 
__global__ void generateBiases(float * biases, numberOfBiases) {
    
}