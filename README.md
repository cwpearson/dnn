# README

This is a DNN written to target CUDA 4.0 (allowing it to run in GPGPUSim).

The architecture is

1. 784 (28 x 28 pixel) input layer
2. [784, 1200] fully-connected layer
3. [1200] sigmoid layer
4. [1200, 100] fully-connected layer
5. [100] sigmoid layer
6. [100,10] fully-connected layer
7. [10] softmax layer

The error function is negative log-liklihood and the training is done through backpropagation with stochastic gradient descent.

The training images are paritioned among the SMs, each of which compute a partial gradient. Those partial gradients are copied back to the host and used to update the network parameters, which are then copied back to the GPU for the next epoch.

The layer implementations can be found in `src/layer`. The data is loaded from `data` in `src/mnist`. The entry point is `src/main.cu`. The MNIST hadwriting dataset is a little bit too challenging for this network, but it seems to partially learn it.

# Prerequisites

CUDA 4.0+

# Building

Adjust `Makefile` to match your system configuration. Then:

    make
    make clean

# Running 

Adjust some parameters at the top of `src/main.cu` if you like, and recompile.

    bin/main