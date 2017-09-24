#ifndef CUDA_RUNTIME_CHECK_HPP
#define CUDA_RUNTIME_CHECK_HPP

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_RUNTIME_CHECK(ans)                                                \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    // fprintf(stderr, "CUDA_CHECK: %s %s %d\n", cudaGetErrorString(code), file,
    //         line);
    std::cerr << "CUDA_CHECK: " << cudaGetErrorString(code) << " " << file
              << " " << line;
    if (abort)
      exit(code);
  }
}
#endif