#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "cuda_runtime_check.hpp"
#include "layer/fc.hu"
#include "layer/sigmoid.hu"
#include "layer/softmax.hu"
#include "mnist.hpp"

const float rate = 0.2;
const int numEpochs = 10000;
const int trainSize = 60000;
const int testSize = 10000;
const int numSMs = 120;
const int minibatchSize = 10;
const bool useTrainForTest = false;

__device__ void block_output_backward(float *errorOut, const float *networkOut,
                                      const int expected, const size_t size) {

  // log/liklihood and softmax error
  for (int o = threadIdx.x; o < size; o += blockDim.x) {
    if (o == expected) {
      errorOut[o] = networkOut[o] - 1;
    } else {
      errorOut[o] = networkOut[o];
    }
  }
}

__device__ void block_network_forward(float *fc1_y, float *r1_y, float *fc2_y,
                                      float *r2_y, float *fc3_y, float *s1_y,
                                      const float *fc1_w, const float *fc1_b,
                                      const float *fc2_w, const float *fc2_b,
                                      const float *fc3_w, const float *fc3_b,
                                      const float *img) {
  block_fc_forward(fc1_y, img, fc1_w, fc1_b, 1200, 784);
  block_sigmoid_forward(r1_y, fc1_y, 1200);
  block_fc_forward(fc2_y, r1_y, fc2_w, fc2_b, 100, 1200);
  block_sigmoid_forward(r2_y, fc2_y, 100);
  block_fc_forward(fc3_y, r2_y, fc3_w, fc3_b, 10, 100);
  block_softmax_forward(s1_y, fc3_y, 10);
}

__global__ void network_forward_kernel(float *fc1_y, float *r1_y, float *fc2_y,
                                       float *r2_y, float *fc3_y, float *s1_y,
                                       const float *fc1_w, const float *fc1_b,
                                       const float *fc2_w, const float *fc2_b,
                                       const float *fc3_w, const float *fc3_b,
                                       const float *img) {
  block_network_forward(fc1_y, r1_y, fc2_y, r2_y, fc3_y, s1_y, fc1_w, fc1_b,
                        fc2_w, fc2_b, fc3_w, fc3_b, img);
}

__global__ void elemwise_plus_equal(float *dst, const float *src, const float alpha, const int size) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = t; i < size; i += gridDim.x * blockDim.x) {
    dst[i] += alpha * src[i];
  }
}

__device__ void block_zero_floats(float *x, const int size) {
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    x[i] = 0.0f;
  }
}

__global__ void zero_kernel(float *x, const int size) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = t; i < size; i += gridDim.x * blockDim.x) {
    x[i] = 0.0f;
  }
}

__global__ void
train_kernel(float *fc1_dw, float *fc1_db, float *fc1_dy, float *r1_dy,
             float *fc2_dw, float *fc2_db, float *fc2_dy, float *r2_dy,
             float *fc3_dw, float *fc3_db, float *fc3_dy, float *s1_dy,
             float *fc1_y, float *r1_y, float *fc2_y, float *r2_y, float *fc3_y,
             float *s1_y, const float *fc1_w, const float *fc1_b,
             const float *fc2_w, const float *fc2_b, const float *fc3_w,
             const float *fc3_b, const float *img, const int *label,
             const size_t numImages) {

  // each threadBlock is a trainer
  const int trainerIdx = blockIdx.x;

  // each trainer (threadblock) has its own work space
  // layer outputs (and inputs)
  fc1_y = &fc1_y[trainerIdx * 1200];
  r1_y = &r1_y[trainerIdx * 1200];
  fc2_y = &fc2_y[trainerIdx * 100];
  r2_y = &r2_y[trainerIdx * 100];
  fc3_y = &fc3_y[trainerIdx * 10];
  s1_y = &s1_y[trainerIdx * 10];

  // parameter updates, error propogation
  fc1_dw = &fc1_dw[trainerIdx * 1200 * 784];
  fc1_db = &fc1_db[trainerIdx * 1200];
  fc1_dy = &fc1_dy[trainerIdx * 1200];
  r1_dy = &r1_dy[trainerIdx * 1200];

  fc2_dw = &fc2_dw[trainerIdx * 100 * 1200];
  fc2_db = &fc2_db[trainerIdx * 100];
  fc2_dy = &fc2_dy[trainerIdx * 100];
  r2_dy = &r2_dy[trainerIdx * 100];

  fc3_dw = &fc3_dw[trainerIdx * 10 * 100];
  fc3_db = &fc3_db[trainerIdx * 10];
  fc3_dy = &fc3_dy[trainerIdx * 10];
  s1_dy = &s1_dy[trainerIdx * 10];

  // zero out gradients before accumulation in train_kernel
  // block_zero_floats(fc1_dw, 1200 * 784);
  // block_zero_floats(fc1_db, 1200);
  // block_zero_floats(fc2_dw, 100 * 1200);
  // block_zero_floats(fc2_db, 100);
  // block_zero_floats(fc3_dw, 10 * 100);
  // block_zero_floats(fc3_db, 10);

  // Split up the images per SM
  const int imgsPerLearner = numImages / gridDim.x;
  // some images won't be covered by this division
  const int numLeftover = numImages - imgsPerLearner * gridDim.x;
  // assign those images to the lowest-indexed learners
  const int myNumImages = imgsPerLearner + (trainerIdx < numLeftover ? 1 : 0);
  // numLeftover lowest-indexed learners have an extra image, so offset start
  // locations to account
  const int imgIdxStart =
      imgsPerLearner * trainerIdx + min(trainerIdx, numLeftover);
  const int imgIdxEnd = imgIdxStart + myNumImages;

  // const int imgIdxEnd =
  //     max(numImages, numImages / gridDim.x * (blockIdx.x + 1));
  for (int imgIdx = imgIdxStart; imgIdx < imgIdxEnd; ++imgIdx) {

    // Have SM collaboratively compute outputs for forward pass
    block_network_forward(fc1_y, r1_y, fc2_y, r2_y, fc3_y, s1_y, fc1_w, fc1_b,
                          fc2_w, fc2_b, fc3_w, fc3_b, &img[imgIdx * 784]);

    // compute error of final layer
    block_output_backward(s1_dy, s1_y, label[imgIdx], 10);
    block_softmax_backward(fc3_dy, s1_dy, s1_y, 10);

    // backprop error
    block_fc_backward(r2_dy, fc3_dw, fc3_db, fc3_dy, r2_y, fc3_w, fc3_b, 10,
                      100);
    block_sigmoid_backward(fc2_dy, r2_dy, r2_y, 100);
    block_fc_backward(r1_dy, fc2_dw, fc2_db, fc2_dy, r1_y, fc2_w, fc2_b, 100,
                      1200);
    block_sigmoid_backward(fc1_dy, r1_dy, r1_y, 1200);
    block_fc_backward(0 /* dont need error at input layer */, fc1_dw, fc1_db,
                      fc1_dy, &img[imgIdx * 784], fc1_w, fc1_b, 1200, 784);
  }
}

void init_weights(float *x, const int size, const int numInputs) {
  for (int i = 0; i < size; ++i) {
    x[i] = ((float(rand()) / RAND_MAX) * 2 - 1) * sqrt(2.0f / numInputs);
  }
}

int argmax(const float *x, const int size) {
  int maxPos = 0;
  float maxVal = -1.0f * INFINITY;

  for (int i = 0; i < size; ++i) {
    if (x[i] > maxVal) {
      maxVal = x[i];
      maxPos = i;
    }
  }
  return maxPos;
}

int main(void) {

  // Read the input data
  float *mnistTestData, *mnistTrainData;
  int *mnistTestLabels, *mnistTrainLabels;
  int mnistNumTestImages, mnistNumTrainImages;

  assert(load_mnist_train(&mnistTrainData, &mnistTrainLabels,
    &mnistNumTrainImages));

  // Load separate test data, or reuse train data
  if (useTrainForTest) {
    mnistTestData = mnistTrainData;
    mnistTestLabels = mnistTrainLabels;
    mnistNumTestImages = mnistNumTrainImages;
  } else {
    assert(
        load_mnist_test(&mnistTestData, &mnistTestLabels, &mnistNumTestImages));
  }


  if (trainSize < mnistNumTrainImages) {
    mnistNumTrainImages = trainSize;
  }
  if (testSize < mnistNumTestImages) {
    mnistNumTestImages = testSize;
  }



  printf("Using %d training images\n", mnistNumTrainImages);
  printf("Using %d test images\n", mnistNumTestImages);

  // Set the CUDA device
  int device = 0;
  CUDA_RUNTIME_CHECK(cudaSetDevice(device));

  // Configure the number of learners
  printf("Using %d learners\n", numSMs);

  // Copy mnist data to the device
  float *imgTrain_d, *imgTest_d;
  int *labelTrain_d;
  CUDA_RUNTIME_CHECK(
      cudaMalloc(&imgTrain_d, 784 * mnistNumTrainImages * sizeof(float)));
  CUDA_RUNTIME_CHECK(
      cudaMalloc(&labelTrain_d, mnistNumTrainImages * sizeof(int)));
  CUDA_RUNTIME_CHECK(
      cudaMalloc(&imgTest_d, 784 * mnistNumTestImages * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMemcpy(imgTrain_d, mnistTrainData,
                                784 * mnistNumTrainImages * sizeof(float),
                                cudaMemcpyHostToDevice));
  CUDA_RUNTIME_CHECK(cudaMemcpy(labelTrain_d, mnistTrainLabels,
                                mnistNumTrainImages * sizeof(int),
                                cudaMemcpyHostToDevice));
  CUDA_RUNTIME_CHECK(cudaMemcpy(imgTest_d, mnistTestData,
                                784 * mnistNumTestImages * sizeof(float),
                                cudaMemcpyHostToDevice));

  // Initialize layer weights and biases
  float *fc1_w_h = new float[784 * 1200]; // layer 1 weights
  float *fc1_b_h = new float[1200];       // layer1 bias
  float *fc2_w_h = new float[1200 * 100];
  float *fc2_b_h = new float[100];
  float *fc3_w_h = new float[100 * 10];
  float *fc3_b_h = new float[10];
  init_weights(fc1_w_h, 784 * 1200, 784);
  init_weights(fc1_b_h, 1200, 1200);
  init_weights(fc2_w_h, 100 * 1200, 1200);
  init_weights(fc2_b_h, 100, 100);
  init_weights(fc3_w_h, 10 * 100, 100);
  init_weights(fc3_b_h, 10, 10);

  // GPU only needs one copy of weights and biases
  float *fc1_w_d, *fc1_b_d, *fc2_w_d, *fc2_b_d, *fc3_w_d, *fc3_b_d;
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc1_w_d, 784 * 1200 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc1_b_d, 1200 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc2_w_d, 1200 * 100 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc2_b_d, 100 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc3_w_d, 100 * 10 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc3_b_d, 10 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMemcpy(fc1_w_d, fc1_w_h, 784 * 1200 * sizeof(float),
                                cudaMemcpyHostToDevice));
  CUDA_RUNTIME_CHECK(cudaMemcpy(fc1_b_d, fc1_b_h, 1200 * sizeof(float),
                                cudaMemcpyHostToDevice));
  CUDA_RUNTIME_CHECK(cudaMemcpy(fc2_w_d, fc2_w_h, 1200 * 100 * sizeof(float),
                                cudaMemcpyHostToDevice));
  CUDA_RUNTIME_CHECK(cudaMemcpy(fc2_b_d, fc2_b_h, 100 * sizeof(float),
                                cudaMemcpyHostToDevice));
  CUDA_RUNTIME_CHECK(cudaMemcpy(fc3_w_d, fc3_w_h, 100 * 10 * sizeof(float),
                                cudaMemcpyHostToDevice));
  CUDA_RUNTIME_CHECK(
      cudaMemcpy(fc3_b_d, fc3_b_h, 10 * sizeof(float), cudaMemcpyHostToDevice));

  // Anything updated by the device needs space per trainer
  float *fc1_y_d;                        // device layer output
  float *fc1_dw_d, *fc1_db_d, *fc1_dy_d; // gradients

  // one copy of gradients, output per trainer
  // FIXME - shared memory?
  float *fc1_dw_h = new float[numSMs * 784 * 1200];
  float *fc1_db_h = new float[numSMs * 1200];
  float *fc1_dy_h = new float[numSMs * 1200];
  float *fc1_y_h = new float[numSMs * 1200];
  CUDA_RUNTIME_CHECK(
      cudaMalloc(&fc1_dw_d, numSMs * 784 * 1200 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc1_db_d, numSMs * 1200 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc1_dy_d, numSMs * 1200 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc1_y_d, numSMs * 1200 * sizeof(float)));

  // relu outputs
  float *r1_y_d, *r1_dy_d;
  float *r1_dy_h = new float[numSMs * 1200 * sizeof(float)];
  float *r1_y_h = new float[numSMs * 1200 * sizeof(float)];
  CUDA_RUNTIME_CHECK(cudaMalloc(&r1_y_d, numSMs * 1200 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&r1_dy_d, numSMs * 1200 * sizeof(float)));

  float *fc2_y_d;
  float *fc2_dw_d, *fc2_db_d, *fc2_dy_d;
  float *fc2_dw_h = new float[numSMs * 1200 * 100];
  float *fc2_db_h = new float[numSMs * 100];
  float *fc2_dy_h = new float[numSMs * 100];
  float *fc2_y_h = new float[numSMs * 100];
  CUDA_RUNTIME_CHECK(
      cudaMalloc(&fc2_dw_d, numSMs * 1200 * 100 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc2_db_d, numSMs * 100 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc2_y_d, numSMs * 100 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc2_dy_d, numSMs * 100 * sizeof(float)));

  float *r2_y_d, *r2_dy_d;
  float *r2_dy_h = new float[numSMs * 100 * sizeof(float)];
  float *r2_y_h = new float[numSMs * 100 * sizeof(float)];
  CUDA_RUNTIME_CHECK(cudaMalloc(&r2_y_d, numSMs * 100 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&r2_dy_d, numSMs * 100 * sizeof(float)));

  float *fc3_y_d;
  float *fc3_dw_d, *fc3_db_d, *fc3_dy_d;
  float *fc3_dw_h = new float[numSMs * 100 * 10];
  float *fc3_db_h = new float[numSMs * 10];
  float *fc3_dy_h = new float[numSMs * 10];
  float *fc3_y_h = new float[numSMs * 10];
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc3_dw_d, numSMs * 100 * 10 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc3_db_d, numSMs * 10 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc3_y_d, numSMs * 10 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&fc3_dy_d, numSMs * 10 * sizeof(float)));

  float *s1_y_d, *s1_dy_d;
  float *s1_y_h = new float[numSMs * 10];
  float *s1_dy_h = new float[numSMs * 10];
  CUDA_RUNTIME_CHECK(cudaMalloc(&s1_y_d, numSMs * 10 * sizeof(float)));
  CUDA_RUNTIME_CHECK(cudaMalloc(&s1_dy_d, numSMs * 10 * sizeof(float)));

  // Do some training
  const int threadsPerBlock = 512;
  dim3 gridDim(numSMs);
  dim3 blockDim(threadsPerBlock);
  for (int epoch = 0; epoch < numEpochs; ++epoch) {

    // check the current quality of the network
    float error = 0.0;
    int numWrong = 0;
    for (int i = 0; i < mnistNumTestImages; ++i) {

      network_forward_kernel<<<1, blockDim>>>(
          fc1_y_d, r1_y_d, fc2_y_d, r2_y_d, fc3_y_d, s1_y_d, fc1_w_d, fc1_b_d,
          fc2_w_d, fc2_b_d, fc3_w_d, fc3_b_d, &imgTest_d[i * 784]);
      CUDA_RUNTIME_CHECK(cudaMemcpy(s1_y_h, s1_y_d, 10 * sizeof(float),
                                    cudaMemcpyDeviceToHost));

      int actual = argmax(s1_y_h, 10);
      const int expected = mnistTestLabels[i];

      error += -1.0f * log(s1_y_h[expected]);
      if (actual != expected) {
        ++numWrong;
      }
    }
    printf("%d %f %f\n", epoch, error / mnistNumTestImages,
           float(numWrong) / mnistNumTestImages);

    // zero out gradient updates before training
    zero_kernel<<<numSMs, blockDim>>>(fc1_dw_d, numSMs * 1200 * 784);
    zero_kernel<<<numSMs, blockDim>>>(fc1_db_d, numSMs * 1200);
    zero_kernel<<<numSMs, blockDim>>>(fc2_dw_d, numSMs * 100 * 1200);
    zero_kernel<<<numSMs, blockDim>>>(fc2_db_d, numSMs * 100);
    zero_kernel<<<numSMs, blockDim>>>(fc3_dw_d, numSMs * 10 * 100);
    zero_kernel<<<numSMs, blockDim>>>(fc3_db_d, numSMs * 10);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    train_kernel<<<gridDim, blockDim>>>(
        fc1_dw_d, fc1_db_d, fc1_dy_d, r1_dy_d, fc2_dw_d, fc2_db_d, fc2_dy_d,
        r2_dy_d, fc3_dw_d, fc3_db_d, fc3_dy_d, s1_dy_d, fc1_y_d, r1_y_d,
        fc2_y_d, r2_y_d, fc3_y_d, s1_y_d, fc1_w_d, fc1_b_d, fc2_w_d, fc2_b_d,
        fc3_w_d, fc3_b_d, imgTrain_d, labelTrain_d, mnistNumTrainImages);

    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());


    // Apply weight updates

    const float alpha = -1 * rate / mnistNumTrainImages;
    for (size_t i = 0; i < numSMs; ++i) {
      // for (size_t j = 0; j < 1200 * 784; ++j) {
      //   fc1_w_h[j] -= fc1_dw_h[i * 1200 * 784 + j] * rate / mnistNumTrainImages;
      // }
      elemwise_plus_equal<<<10, 512>>>(fc1_w_d, &fc1_dw_d[i * 784 * 1200], alpha, 784 * 1200);
      // for (size_t j = 0; j < 1200; ++j) {
      //   fc1_b_h[j] -= fc1_db_h[i * 1200 + j] * rate / mnistNumTrainImages;
      // }
      elemwise_plus_equal<<<10, 512>>>(fc1_b_d, &fc1_db_d[i * 1200], alpha, 1200);
      // for (size_t j = 0; j < 100 * 1200; ++j) {
      //   fc2_w_h[j] -= fc2_dw_h[i * 100 * 1200 + j] * rate / mnistNumTrainImages;
      // }
      elemwise_plus_equal<<<10, 512>>>(fc2_w_d, &fc2_dw_d[i * 1200 * 100], alpha, 1200 * 100);
      // for (size_t j = 0; j < 100; ++j) {
      //   fc2_b_h[j] -= fc2_db_h[i * 100 + j] * rate / mnistNumTrainImages;
      // }
      elemwise_plus_equal<<<10, 512>>>(fc2_b_d, &fc2_db_d[i * 100], alpha, 100);
      // for (size_t j = 0; j < 10 * 100; ++j) {
      //   fc3_w_h[j] -= fc3_dw_h[i * 10 * 100 + j] * rate / mnistNumTrainImages;
      // }
      elemwise_plus_equal<<<10, 100>>>(fc3_w_d, &fc3_dw_d[i * 10 * 100], alpha, 10 * 100);
      // for (size_t j = 0; j < 10; ++j) {
      //   fc3_b_h[j] -= fc3_db_h[i * 10 + j] * rate / mnistNumTrainImages;
      // }
      elemwise_plus_equal<<<10, 512>>>(fc3_b_d, &fc3_db_d[i * 10], alpha, 10);
    }


  }

  return 0;
}
