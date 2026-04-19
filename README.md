# OpenMP Image Convolution

This repository contains the source code for a parallel implementation of a 2D image convolution program using OpenMP in C.

## Description

The program reads a PPM image, reads a convolution kernel from a text file, applies the convolution to the three color channels (R, G, B), and stores the resulting filtered image in an output PPM file.

The implementation uses OpenMP to parallelize the most computationally expensive part of the program: the convolution operation.

## Parallelization decisions

The starting point was the analysis of the sequential version of the program. The most expensive section is the convolution process, where each output pixel is computed by applying the kernel over the corresponding neighborhood of the input image.

Since each output pixel can be computed independently, this part is well suited for data parallelism.

### Main decisions taken

1. **Parallelization of the outer loop**
   The loop over the image rows in the convolution function was parallelized using OpenMP:
   ```c
   #pragma omp parallel for schedule(static)
