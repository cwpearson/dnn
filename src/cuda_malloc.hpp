#ifndef CUDA_MALLOC_HPP
#define CUDA_MALLOC_HPP

#include "cuda_runtime_check.hpp"

template <typename T> 
T *cudaMalloc1D(const size_t d1) {
  T *p;
  CUDA_RUNTIME_CHECK(cudaMalloc(&p, sizeof(T) * d1));
  return p;
}

template <typename T>
T *cudaMalloc3D(const size_t d1, const size_t d2, const size_t d3) {
  T *p;
  CUDA_RUNTIME_CHECK(cudaMalloc(&p, sizeof(T) * d1 * d2 * d3));
  return p;
}

template <typename T>
T *cudaMalloc4D(const size_t d1, const size_t d2, const size_t d3, const size_t d4) {
  T *p;
  CUDA_RUNTIME_CHECK(cudaMalloc(&p, sizeof(T) * d1 * d2 * d3 * d4));
  return p;
}

#endif