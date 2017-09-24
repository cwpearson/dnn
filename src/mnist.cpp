#include "mnist.hpp"

#include <cstdio>
#include <cstdlib>

float rand01() { return float(rand()) / RAND_MAX; }

void make_image(float *data, const int label) {
  for (int i = 0; i < 784; ++i) {
    if (i > label * 784 / 10 && i < (label + 1) * 784 / 10) {
      data[i] = 0.5;
    } else {
      data[i] = -0.5;
    }
  }
}

int from_big_endian(unsigned char buf[4]) {
  return (int)buf[3] | (int)buf[2] << 8 | (int)buf[1] << 16 | (int)buf[0] << 24;
}

int fake_mnist(float **data, int **labels, int numImages) {
  *data = new float[784 * numImages];
  *labels = new int[numImages];

  for (int i = 0; i < numImages; ++i) {
    const int label = i % 2;
    (*labels)[i] = label;
    make_image(&(*data)[i * 784], label);
  }
  return 1;
}

int load_images(float **data, int *numImages, const char *path) {
  FILE *f = fopen(path, "r");
  if (NULL != f) {
    fseek(f, 4, SEEK_SET); // skip magic number

    unsigned char buf[4];

    // read number of items
    if (4 != fread(buf, 1, 4, f)) {
      return 0;
    }
    *numImages = from_big_endian(buf);
    printf("Loading %d images\n", *numImages);
    *data = new float[*numImages * 28 * 28];

    // skip number of rows and columns
    fseek(f, 8, SEEK_CUR);

    // read the image data and scale to 0-1
    // FIXME - don't need to do this one byte at a time
    for (int i = 0; i < *numImages; ++i) {
      for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
          if (1 != fread(buf, 1, 1, f)) {
            return 0;
          }
          (*data)[i * 784 + y * 28 + x] = buf[0] / 256.0f;
        }
      }
    }

    // Subtract out average
    for (int i = 0; i < *numImages; ++i) {
      float avg = 0.0;
      for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
          avg += (*data)[i * 784 + y * 28 + x];
        }
      }
      avg /= 28 * 28;
      for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
          (*data)[i * 784 + y * 28 + x] -= avg;
        }
      }
    }

    fclose(f);
    return 1;
  }
  return 0;
}

int load_labels(int **labels, int *numImages, const char *path) {
  FILE *f = fopen(path, "r");
  if (NULL != f) {
    fseek(f, 4, SEEK_SET); // skip magic number
    unsigned char buf[4];

    // read number of items
    if (4 != fread(buf, 1, 4, f)) {
      return 0;
    }
    *numImages = from_big_endian(buf);
    printf("Loading %d labels\n", *numImages);
    *labels = new int[*numImages];

    // read the labels
    for (int i = 0; i < *numImages; ++i) {
      if (1 != fread(buf, 1, 1, f)) {
        return 0;
      }
      (*labels)[i] = buf[0];
    }

    fclose(f);
    return 1;
  }
  return 0;
}

int load_mnist_train(float **data, int **labels, int *numImages) {
  int success = load_labels(labels, numImages, "data/train-labels-idx1-ubyte");
  if (!success) {
    return success;
  }
  return load_images(data, numImages, "data/train-images-idx3-ubyte");
}

int load_mnist_test(float **data, int **labels, int *numImages) {
  int success = load_labels(labels, numImages, "data/t10k-labels-idx1-ubyte");
  if (!success) {
    return success;
  }
  return load_images(data, numImages, "data/t10k-images-idx3-ubyte");
}
