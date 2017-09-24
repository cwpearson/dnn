#ifndef MNIST_HPP
#define MNIST_HPP

int load_mnist_train(float **data, int **labels, int *numImages);
int load_mnist_test(float **data, int **labels, int *numImages);

#endif