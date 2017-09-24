# README

This is a DNN written to target CUDA 4.0 (allowing it to run in GPGPUSim).

# Prerequisites

CUDA 4.0

# Building

Adjust `Makefile` to match your system configuration. Then:

    make
    make clean

# Running 

Adjust some parameters at the top of `src/main.cu` if you like, and recompile.

    bin/main

