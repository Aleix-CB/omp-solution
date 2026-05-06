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
   ```

---

# MPI + OpenMP Image Convolution

This repository also includes a hybrid implementation using MPI and OpenMP.

## Description

The program reads a PPM image, reads a convolution kernel from a text file, applies the convolution to the three color channels (R, G, B), and stores the resulting filtered image in an output PPM file.

The implementation combines:
- **MPI** to distribute work between processes  
- **OpenMP** to parallelize computation within each process  

## Parallelization decisions

The starting point was the same sequential analysis: the convolution is the most computationally expensive part of the program.

Since each pixel and each color channel can be computed independently, the problem is well suited for parallelization.

In this implementation, parallelism is exploited at two levels:
- Process-level parallelism (MPI)  
- Thread-level parallelism (OpenMP)  

### Main decisions taken

1. **Distribution of work using MPI**  
   A master-worker model is used:

   - **Master (rank 0)**:
     - Reads the image and kernel  
     - Sends each color channel (R, G, B) to a different worker  
     - Receives processed data  
     - Writes the final image  

   - **Workers (rank 1–3)**:
     - Each process handles one color channel  
     - Performs convolution  
     - Sends results back  

2. **Parallelization of the outer loop (OpenMP)**  
   Inside each worker, the loop over the image rows in the convolution function is parallelized using OpenMP:
   ```c
   #pragma omp parallel for schedule(static)
   ```

3. **Effective use of cores**  
   The program is executed with:
   - 4 MPI processes (1 master + 3 workers)  
   - 4 OpenMP threads per worker  

   However, the master process only uses 1 core.

   Therefore:
   - Total theoretical cores: 16  
   - **Effective cores used: 13 (3 × 4 + 1)**  

   This must be taken into account when computing performance metrics such as speedup and efficiency.
