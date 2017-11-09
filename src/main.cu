#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector>

#include "cuda_runtime_check.hpp"
#include "cuda_malloc.hpp"
#include "blas/saxpy.hu"
#include "layer/fc.hu"
#include "layer/sigmoid.hu"
#include "layer/softmax.hu"
#include "mnist.hpp"

const float rate = 0.2;
const int numEpochs = 10000;
const int trainSize = 6000;
const int testSize = 1000;
const int numSMs = 10;
const int maxBatchSize = 1;
const bool useTrainForTest = false;
const int cudaDevice = 0;


#define Array3D(_ptr, _i, _j, _k, _d_i, _d_j) (_ptr[_i * (_d_i * _d_j) + _j * _d_j + _k])

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


__device__ void output_backward_batch_block(float *errorOut, const float *networkOut,
  const int *expected, const size_t size, const int batchSize) {

  #define EO(b, o) (errorOut[b * size + o])
  #define NO(b, o) (networkOut[b * size + o])
  #define E(b) (expected[b])

  for (int b = 0; b < batchSize; ++b) {
    // log/liklihood and softmax error
    for (int o = threadIdx.x; o < size; o += blockDim.x) {
      if (o == E(b)) {
        EO(b, o) = NO(b, 0) - 1;
      } else {
        EO(b, o) = NO(b, 0);
      }
    }
  }

  #undef E
  #undef NO
  #undef EO
}


__device__ void network_forward_block(float *fc1_y, float *r1_y, float *fc2_y,
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

__device__ void network_forward_batch_block(float *fc1_y, float *r1_y, float *fc2_y,
  float *r2_y, float *fc3_y, float *s1_y,
  const float *fc1_w, const float *fc1_b,
  const float *fc2_w, const float *fc2_b,
  const float *fc3_w, const float *fc3_b,
  const float *img, const size_t batchSize) {
  fc_forward_batch_block(fc1_y, img, fc1_w, fc1_b, 1200, 784, batchSize);
  sigmoid_forward_batch_block(r1_y, fc1_y, 1200, batchSize);
  fc_forward_batch_block(fc2_y, r1_y, fc2_w, fc2_b, 100, 1200, batchSize);
  sigmoid_forward_batch_block(r2_y, fc2_y, 100, batchSize);
  fc_forward_batch_block(fc3_y, r2_y, fc3_w, fc3_b, 10, 100, batchSize);
  softmax_forward_batch_block(s1_y, fc3_y, 10, batchSize);
}

__global__ void network_forward_batch_kernel(float *fc1_y, float *r1_y, float *fc2_y,
  float *r2_y, float *fc3_y, float *s1_y,
  const float *fc1_w, const float *fc1_b,
  const float *fc2_w, const float *fc2_b,
  const float *fc3_w, const float *fc3_b,
  const float *img, const size_t batchSize) {
  network_forward_batch_block(fc1_y, r1_y, fc2_y, r2_y, fc3_y, s1_y, fc1_w, fc1_b,
                              fc2_w, fc2_b, fc3_w, fc3_b, img, batchSize);
}

__global__ void network_forward_kernel(float *fc1_y, float *r1_y, float *fc2_y,
                                       float *r2_y, float *fc3_y, float *s1_y,
                                       const float *fc1_w, const float *fc1_b,
                                       const float *fc2_w, const float *fc2_b,
                                       const float *fc3_w, const float *fc3_b,
                                       const float *img) {
  network_forward_block(fc1_y, r1_y, fc2_y, r2_y, fc3_y, s1_y, fc1_w, fc1_b,
                        fc2_w, fc2_b, fc3_w, fc3_b, img);
}

// zero_block uses a block of threads to collaboratively zero a pointer
__device__ void zero_block(float *x, const int size) {
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    x[i] = 0.0f;
  }
}

// zero_block uses a grid of threads to collaboratively zero a pointer
__global__ void zero_grid(float *x, const int size) {
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

  for (int imgIdx = imgIdxStart; imgIdx < imgIdxEnd; ++imgIdx) {

    // Have SM collaboratively compute outputs for forward pass
    network_forward_block(fc1_y, r1_y, fc2_y, r2_y, fc3_y, s1_y, fc1_w, fc1_b,
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


__global__ void
train_batch_kernel(float *fc1_dw, float *fc1_db, float *fc1_dy, float *r1_dy,
             float *fc2_dw, float *fc2_db, float *fc2_dy, float *r2_dy,
             float *fc3_dw, float *fc3_db, float *fc3_dy, float *s1_dy,
             float *fc1_y, float *r1_y, float *fc2_y, float *r2_y, float *fc3_y,
             float *s1_y, const float *fc1_w, const float *fc1_b,
             const float *fc2_w, const float *fc2_b, const float *fc3_w,
             const float *fc3_b,
             const float * const*perLearnerImg, const int * const*perLearnerLabel, const int *perLearnerBatchSize) {

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

  const float *img = perLearnerImg[trainerIdx]; // batch image data
  const int *label = perLearnerLabel[trainerIdx]; // batch label data
  const int batchSize = perLearnerBatchSize[trainerIdx]; // size of batch

  // Have SM collaboratively compute outputs for forward pass
  network_forward_batch_block(fc1_y, r1_y, fc2_y, r2_y, fc3_y, s1_y, fc1_w, fc1_b,
                              fc2_w, fc2_b, fc3_w, fc3_b, img, batchSize);


  // Zero out the weight updates
  zero_block(fc1_dw, maxBatchSize * 1200 * 784);
  zero_block(fc1_db, maxBatchSize * 784);
  zero_block(fc2_dw, maxBatchSize * 100 * 1200);
  zero_block(fc2_db, maxBatchSize * 1200);
  zero_block(fc3_dw, maxBatchSize * 10 * 100);
  zero_block(fc3_db, maxBatchSize * 10);

  // compute error of final layer
  output_backward_batch_block(s1_dy, s1_y, label, 10, batchSize);
  softmax_backward_batch_block(fc3_dy, s1_dy, s1_y, 10, batchSize);

  // backprop error
  fc_backward_batch_block(r2_dy, fc3_dw, fc3_db, fc3_dy, r2_y, fc3_w, fc3_b, 10,
                          100, batchSize);
  sigmoid_backward_batch_block(fc2_dy, r2_dy, r2_y, 100, batchSize);
  fc_backward_batch_block(r1_dy, fc2_dw, fc2_db, fc2_dy, r1_y, fc2_w, fc2_b, 100,
                          1200, batchSize);
  sigmoid_backward_batch_block(fc1_dy, r1_dy, r1_y, 1200, batchSize);
  fc_backward_batch_block(0 /* dont need error at input layer */, fc1_dw, fc1_db,
                          fc1_dy, img, fc1_w, fc1_b, 1200, 784, batchSize);
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

  printf("Using batch size: %d\n", maxBatchSize);

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

  // Reduce the training and test size, if desired
  if (trainSize < mnistNumTrainImages) {
    mnistNumTrainImages = trainSize;
  }
  if (testSize < mnistNumTestImages) {
    mnistNumTestImages = testSize;
  }


  printf("Using %d training images\n", mnistNumTrainImages);
  printf("Using %d test images\n", mnistNumTestImages);

  CUDA_RUNTIME_CHECK(cudaSetDevice(cudaDevice));

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

  // a pointer to each learner's image data and labels for each batch
  std::vector<float *> perLearnerImg_h(numSMs);
  std::vector<int *> perLearnerLabel_h(numSMs);
  std::vector<int> perLearnerBatchSize_h(numSMs);
  float **perLearnerImg_d = cudaMalloc1D<float*>(numSMs);
  int **perLearnerLabel_d = cudaMalloc1D<int*>(numSMs);
  int *perLearnerBatchSize_d = cudaMalloc1D<int>(numSMs);


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

  // GPU only needs one copy of model weights and biases
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

  // Anything updated by the device needs space per trainer and per minibatch
  // one copy of gradients, output per trainer
  // FIXME - shared memory?
  float *fc1_dw_h = new float[maxBatchSize * numSMs * 784 * 1200];
  float *fc1_db_h = new float[maxBatchSize * numSMs * 1200];
  float *fc1_dy_h = new float[maxBatchSize * numSMs * 1200];
  float *fc1_y_h  = new float[maxBatchSize * numSMs * 1200];
  float *fc1_y_d =  cudaMalloc4D<float>(numSMs, maxBatchSize, 784, 1200); // device layer output
  float *fc1_dw_d = cudaMalloc4D<float>(numSMs, maxBatchSize, 784, 1200); // gradients
  float *fc1_dy_d = cudaMalloc3D<float>(numSMs, maxBatchSize, 1200);
  float *fc1_db_d = cudaMalloc3D<float>(numSMs, maxBatchSize, 1200);

  // relu outputs
  float *r1_dy_h = new float[maxBatchSize * numSMs * 1200 * sizeof(float)];
  float *r1_y_h = new float[maxBatchSize * numSMs * 1200 * sizeof(float)];
  float *r1_dy_d = cudaMalloc3D<float>(numSMs, maxBatchSize, 1200);
  float *r1_y_d  = cudaMalloc3D<float>(numSMs, maxBatchSize, 1200);

  // fc2 outputs
  float *fc2_dw_d = cudaMalloc4D<float>(numSMs, maxBatchSize, 1200, 100);
  float *fc2_db_d = cudaMalloc3D<float>(numSMs, maxBatchSize, 100);
  float *fc2_dy_d = cudaMalloc3D<float>(numSMs, maxBatchSize, 100);
  float *fc2_y_d  = cudaMalloc3D<float>(numSMs, maxBatchSize, 100);
  float *fc2_dw_h = new float[maxBatchSize * numSMs * 1200 * 100];
  float *fc2_db_h = new float[maxBatchSize * numSMs * 100];
  float *fc2_dy_h = new float[maxBatchSize * numSMs * 100];
  float *fc2_y_h =  new float[maxBatchSize * numSMs * 100];

  float *r2_dy_d = cudaMalloc3D<float>(numSMs, maxBatchSize, 100);
  float *r2_y_d =  cudaMalloc3D<float>(numSMs, maxBatchSize, 100);
  float *r2_dy_h = new float[maxBatchSize * numSMs * 100 * sizeof(float)];
  float *r2_y_h =  new float[maxBatchSize * numSMs * 100 * sizeof(float)];

  float *fc3_dw_d = cudaMalloc4D<float>(numSMs, maxBatchSize, 100, 10); 
  float *fc3_db_d = cudaMalloc3D<float>(numSMs, maxBatchSize, 10); 
  float *fc3_dy_d = cudaMalloc3D<float>(numSMs, maxBatchSize, 10);
  float *fc3_y_d  = cudaMalloc3D<float>(numSMs, maxBatchSize, 10);
  float *fc3_dw_h = new float[maxBatchSize * numSMs * 100 * 10];
  float *fc3_db_h = new float[maxBatchSize * numSMs * 10];
  float *fc3_dy_h = new float[maxBatchSize * numSMs * 10];
  float *fc3_y_h =  new float[maxBatchSize * numSMs * 10];

  float *s1_y_d  = cudaMalloc3D<float>(numSMs, maxBatchSize, 10);
  float *s1_dy_d = cudaMalloc3D<float>(numSMs, maxBatchSize, 10);
  float *s1_y_h =  new float[maxBatchSize * numSMs * 10];
  float *s1_dy_h = new float[maxBatchSize * numSMs * 10];

  // Do some training
  const int threadsPerBlock = 512;
  dim3 gridDim(numSMs);
  dim3 blockDim(threadsPerBlock);
  for (int epoch = 0; epoch < numEpochs; ++epoch) {

    // check the current quality of the network
    float error = 0.0;
    int numWrong = 0;
    for (int batchStart = 0; batchStart < mnistNumTestImages; batchStart += maxBatchSize) {
      const int batchEnd = min(mnistNumTestImages, batchStart + maxBatchSize);
      const int batchSize = batchEnd - batchStart;

      const float *imgTestBatch_d = &imgTest_d[batchStart * 784];

      network_forward_batch_kernel<<<1, blockDim>>>(
        fc1_y_d, r1_y_d, fc2_y_d, r2_y_d, fc3_y_d, s1_y_d, fc1_w_d, fc1_b_d,
        fc2_w_d, fc2_b_d, fc3_w_d, fc3_b_d, imgTestBatch_d, batchSize);

      // copy network output back to host
      CUDA_RUNTIME_CHECK(cudaMemcpy(s1_y_h, s1_y_d, batchSize * 10 * sizeof(float), cudaMemcpyDeviceToHost));
      for (int i = 0; i < batchSize; ++i) {
        int actual = argmax(&s1_y_h[i * 10], 10);
        const int expected = mnistTestLabels[batchStart + i];
        error += -1.0f * log(s1_y_h[i * 10 + expected]);
        if (actual != expected) {
          ++numWrong;
        }
      }
    }

    printf("epoch=%d error=%f accuracy=%f (%d wrong / %d total)\n", epoch, error / mnistNumTestImages,
           float(numWrong) / mnistNumTestImages, numWrong, mnistNumTestImages);

    // Do training
    for (int imgIdx = 0; imgIdx < mnistNumTrainImages; ) {
      // Determine where each learner's batch begins
      for (int learnerIdx = 0; learnerIdx < numSMs; ++learnerIdx) {
        perLearnerImg_h[learnerIdx] = &imgTrain_d[imgIdx * 784];
        perLearnerLabel_h[learnerIdx] = &labelTrain_d[imgIdx];
        int imgsLeft = mnistNumTrainImages - imgIdx;
        const int batchSize = max(0 , min(maxBatchSize, imgsLeft));
        perLearnerBatchSize_h[learnerIdx] = batchSize;
        imgIdx += batchSize;
        assert(imgIdx <= mnistNumTrainImages);
      }

      // copy that data to GPU
      CUDA_RUNTIME_CHECK(cudaMemcpy(perLearnerImg_d, &perLearnerImg_h[0], sizeof(float *) * numSMs, cudaMemcpyHostToDevice));
      CUDA_RUNTIME_CHECK(cudaMemcpy(perLearnerLabel_d, &perLearnerLabel_h[0], sizeof(int *) * numSMs, cudaMemcpyHostToDevice));
      CUDA_RUNTIME_CHECK(cudaMemcpy(perLearnerBatchSize_d, &perLearnerBatchSize_h[0], sizeof(int) * numSMs, cudaMemcpyHostToDevice));

    train_batch_kernel<<<gridDim, blockDim>>>(
        fc1_dw_d, fc1_db_d, fc1_dy_d, r1_dy_d, fc2_dw_d, fc2_db_d, fc2_dy_d,
        r2_dy_d, fc3_dw_d, fc3_db_d, fc3_dy_d, s1_dy_d, fc1_y_d, r1_y_d,
        fc2_y_d, r2_y_d, fc3_y_d, s1_y_d, fc1_w_d, fc1_b_d, fc2_w_d, fc2_b_d,
        fc3_w_d, fc3_b_d, perLearnerImg_d, perLearnerLabel_d, perLearnerBatchSize_d);
    CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());

    // Apply weight updates. Each learner and each image in a batch produces its own set of updates
    const float alpha = -1 * rate / mnistNumTrainImages;
    for (size_t i = 0; i < numSMs; ++i) {
      for (size_t j = 0; j < maxBatchSize; ++j) {
      saxpy_grid<<<(784 * 1200 + 255) / 256, 256>>>(fc1_w_d, &fc1_dw_d[(i * maxBatchSize + j) * 784 * 1200], alpha, 784 * 1200);
      saxpy_grid<<<(1200 + 255) / 256, 256>>>(fc1_b_d, &fc1_db_d[(i * maxBatchSize + j) * 1200], alpha, 1200);
      saxpy_grid<<<(1200 * 100 + 255) / 256, 256>>>(fc2_w_d, &fc2_dw_d[(i * maxBatchSize + j) * 1200 * 100], alpha, 1200 * 100);
      saxpy_grid<<<(100 + 31) / 32, 32>>>(fc2_b_d, &fc2_db_d[(i * maxBatchSize + j) * 100], alpha, 100);
      saxpy_grid<<<(10 * 100 + 127) / 128, 128>>>(fc3_w_d, &fc3_dw_d[(i * maxBatchSize + j) * 10 * 100], alpha, 10 * 100);
      saxpy_grid<<<1, 32>>>(fc3_b_d, &fc3_db_d[(i * maxBatchSize + j) * 10], alpha, 10);
      }
    }
  }
}

  return 0;
}
