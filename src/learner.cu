#include "learner.hpp"

__global__ void learnkern(void *param) { return; }

void learn() { learnkern<<<1, 1>>>(0); }