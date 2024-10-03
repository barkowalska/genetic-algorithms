#include <cross_over.cuh>

__global__ void cross_over(bool* chromosomA, bool* chromosomB, int n)
{
    unsigned int i=blockDim.x*blockIdx.x+threadIdx.x;
    
    int tmp=chromosomA[i];
    chromosomA[i]=chromosomB[i];
    chromosomB[i]=tmp;

}