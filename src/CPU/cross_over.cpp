#include "cross_over.h"
using namespace std;


void cross_over(std::span<bool> chromosomA, std::span<bool> chromosonB)
{
    srand(time(NULL));
    int length=chromosomA.size();
    int point=rand()%length;
    cout<<"point "<<point<<endl; 
    for(int i=point; i<length; i++)
    {
        swap(chromosomA[i], chromosonB[i]);
    }   
    
    
}

