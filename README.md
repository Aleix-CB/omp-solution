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



---

# CUDA Image Convolution

This repository also includes a CUDA implementation of the image convolution algorithm, designed to exploit the massive parallelism available in modern GPUs.

## Description

The program reads a PPM image, reads a convolution kernel from a text file, applies the convolution to the three color channels (R, G, B), and stores the resulting filtered image in an output PPM file.

The implementation uses CUDA to accelerate the convolution process by executing thousands of threads in parallel on one or more GPUs.

The implementation supports both:

* **Single-GPU execution**
* **Multi-GPU execution (2 GPUs)**

## Parallelization decisions

The starting point was the same sequential analysis used in the OpenMP and MPI implementations. The convolution operation represents the dominant computational workload and can be parallelized efficiently because each output pixel can be computed independently.

### Main decisions taken

1. **One thread per output pixel**

   Each CUDA thread is responsible for computing a single output pixel. A two-dimensional grid is used to naturally map CUDA threads to image coordinates.

   ```cpp
   dim3 blockSize(16,16);
   dim3 gridSize((width + 15)/16,
                 (height + 15)/16);
   ```

2. **Use of constant memory for the convolution kernel**

   The convolution kernel is stored in CUDA constant memory.

   Since all threads repeatedly access the same kernel coefficients, constant memory provides an efficient broadcast mechanism and reduces memory traffic compared to storing the kernel in global memory.

3. **Image data stored in global memory**

   The image is stored in global memory.

   Shared memory was evaluated but discarded because the largest convolution kernels (99×99) would require more memory than available per CUDA block.

   For example:

   * Block size: 16×16 threads
   * Halo size for a 99×99 kernel: 49 pixels per side
   * Required tile size:

   ```text
   (16 + 49 + 49) × (16 + 49 + 49)
   = 114 × 114 pixels
   ```

   Considering three RGB channels stored as integers:

   ```text
   114 × 114 × 3 × 4 bytes
   = 155,952 bytes
   ≈ 152.3 KB
   ```

   The NVIDIA RTX A4000 provides:

   * 48 KB shared memory per block
   * 100 KB shared memory per SM

   Therefore, the complete tile cannot fit in shared memory.

4. **Use of GPU cache hierarchy**

   Since shared memory was not suitable for the largest kernels, the implementation relies on:

   * L1 cache
   * L2 cache
   * Constant memory

   to reduce global memory access costs.

5. **Multi-GPU workload distribution**

   For the two-GPU implementation, the image is divided into horizontal regions.

   Each GPU receives approximately half of the image rows:

   ```text
   GPU 0 → first half
   GPU 1 → second half
   ```

   Each device performs the convolution independently and the results are merged at the end of the execution.

6. **Minimal synchronization**

   No synchronization is required during the convolution itself because output pixels are independent.

   Synchronization is only required when:

   * Transferring data to the GPUs
   * Gathering results from the GPUs
   * Writing the final image

## Hardware Used

The CUDA implementation was developed and evaluated on the following workstation:

* 2 × NVIDIA RTX A4000 (16 GB GDDR6)
* 48 Streaming Multiprocessors (SMs) per GPU
* 6144 CUDA Cores per GPU
* Dual Intel Xeon Silver 4210R
* CUDA Toolkit 12.x

## Execution Modes

### Single GPU

```bash
./cuda_rgb_conv input.ppm kernel.txt output.ppm 1
```

### Two GPUs

```bash
./cuda_rgb_conv input.ppm kernel.txt output.ppm 2
```

The last parameter specifies the number of GPUs to use.

## Performance Notes

The CUDA implementation achieves the best performance among all developed versions.

For large images and kernels, speedups between approximately 100× and 175× were obtained when compared with the serial implementation.

The multi-GPU version achieves near-linear scaling for the largest workloads, reaching speedups close to 2× when comparing kernel execution times between one and two GPUs.

However, total application performance is strongly influenced by image I/O operations (reading and writing PPM files), which become the main bottleneck once the convolution itself is accelerated.

   
