/*
 * cuda_convolution.cu
 *
 * Tasca 4 - Convolucio RGB amb CUDA
 *
 * Estrategia utilitzada:
 *   - La imatge sencera es copia a la memoria global de cada GPU.
 *   - El kernel de convolucio es copia a constant memory.
 *   - Cada thread CUDA calcula un pixel complet RGB.
 *   - No s'utilitza shared memory per evitar reduir massa l'occupancy amb kernels grans (fins a 99x99).
 *   - Si hi ha mes d'una GPU disponible, la imatge es reparteix per files.
 *
 * Compilacio:
 *   nvcc -O3 cuda_convolution.cu -o cuda_rgb_conv
 *
 * Execucio:
 *   ./cuda_rgb_conv image.ppm kernel.txt output.ppm
 *
 * Opcionalment es pot limitar el nombre de GPUs:
 *   ./cuda_rgb_conv image.ppm kernel.txt output.ppm 1
 *   ./cuda_rgb_conv image.ppm kernel.txt output.ppm 2
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define MAX_KERNEL_SIZE 99
#define MAX_KERNEL_ELEMS (MAX_KERNEL_SIZE * MAX_KERNEL_SIZE)
#define BLOCK_X 16
#define BLOCK_Y 16

/* Kernel de convolucio a constant memory.
 * 99x99 floats = 9801 * 4 bytes = 39204 bytes, per sota dels 64 KB habituals.
 */
__constant__ float d_kernel_const[MAX_KERNEL_ELEMS];

typedef struct {
    int ancho;
    int altura;
    int maxcolor;
    int *R;
    int *G;
    int *B;
} Imagen;

typedef struct {
    int kx;
    int ky;
    float *data;
} Kernel;

static void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static double nowSeconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline int clampHost(int v, int maxcolor) {
    if (v < 0) return 0;
    if (v > maxcolor) return maxcolor;
    return v;
}

__device__ __forceinline__ int clampDevice(int v, int maxcolor) {
    if (v < 0) return 0;
    if (v > maxcolor) return maxcolor;
    return v;
}

Imagen *allocImage(int w, int h) {
    Imagen *img = (Imagen *)malloc(sizeof(Imagen));
    if (!img) return NULL;

    long long size = (long long)w * (long long)h;

    img->ancho = w;
    img->altura = h;
    img->maxcolor = 255;

    img->R = (int *)malloc(size * sizeof(int));
    img->G = (int *)malloc(size * sizeof(int));
    img->B = (int *)malloc(size * sizeof(int));

    if (!img->R || !img->G || !img->B) {
        free(img->R);
        free(img->G);
        free(img->B);
        free(img);
        return NULL;
    }

    return img;
}

void freeImage(Imagen *img) {
    if (!img) return;
    free(img->R);
    free(img->G);
    free(img->B);
    free(img);
}

void freeKernel(Kernel *k) {
    if (!k) return;
    free(k->data);
    free(k);
}

/* Lectura PPM P3 basada en openMP-MPI2.c */
Imagen *readPPM(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening image");
        return NULL;
    }

    char format[3];
    if (fscanf(fp, "%2s", format) != 1) {
        fclose(fp);
        return NULL;
    }

    if (strcmp(format, "P3") != 0) {
        fprintf(stderr, "Only P3 format supported\n");
        fclose(fp);
        return NULL;
    }

    int c = fgetc(fp);
    while (c == '\n' || c == ' ' || c == '\t' || c == '\r') {
        c = fgetc(fp);
    }

    while (c == '#') {
        while (c != '\n' && c != EOF) {
            c = fgetc(fp);
        }
        c = fgetc(fp);
        while (c == '\n' || c == ' ' || c == '\t' || c == '\r') {
            c = fgetc(fp);
        }
    }

    ungetc(c, fp);

    int w, h, maxcolor;
    if (fscanf(fp, "%d %d", &w, &h) != 2) {
        fclose(fp);
        return NULL;
    }
    if (fscanf(fp, "%d", &maxcolor) != 1) {
        fclose(fp);
        return NULL;
    }

    Imagen *img = allocImage(w, h);
    if (!img) {
        fclose(fp);
        return NULL;
    }
    img->maxcolor = maxcolor;

    long long size = (long long)w * (long long)h;
    for (long long i = 0; i < size; i++) {
        if (fscanf(fp, "%d %d %d", &img->R[i], &img->G[i], &img->B[i]) != 3) {
            fprintf(stderr, "Error reading pixel %lld\n", i);
            freeImage(img);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    return img;
}

void writePPM(const char *filename, Imagen *img) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error writing image");
        return;
    }

    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", img->ancho, img->altura);
    fprintf(fp, "%d\n", img->maxcolor);

    long long size = (long long)img->ancho * (long long)img->altura;
    for (long long i = 0; i < size; i++) {
        fprintf(fp, "%d %d %d ", img->R[i], img->G[i], img->B[i]);
        if ((i + 1) % img->ancho == 0) {
            fprintf(fp, "\n");
        }
    }

    fclose(fp);
}

Kernel *readKernel(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Kernel error");
        return NULL;
    }

    Kernel *k = (Kernel *)malloc(sizeof(Kernel));
    if (!k) {
        fclose(fp);
        return NULL;
    }

    /* Lector tolerant: accepta comes, espais, tabuladors i salts de linia. */
    if (fscanf(fp, " %d%*[, \t\r\n]%d", &k->kx, &k->ky) != 2) {
        fprintf(stderr, "Error reading kernel size\n");
        free(k);
        fclose(fp);
        return NULL;
    }

    if (k->kx <= 0 || k->ky <= 0 || k->kx > MAX_KERNEL_SIZE || k->ky > MAX_KERNEL_SIZE) {
        fprintf(stderr, "Kernel size not supported. Max supported: %dx%d\n",
                MAX_KERNEL_SIZE, MAX_KERNEL_SIZE);
        free(k);
        fclose(fp);
        return NULL;
    }

    int size = k->kx * k->ky;
    k->data = (float *)malloc(size * sizeof(float));
    if (!k->data) {
        free(k);
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < size; i++) {
        int c;
        do {
            c = fgetc(fp);
        } while (c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == ',');

        if (c != EOF) {
            ungetc(c, fp);
        }

        if (fscanf(fp, " %f", &k->data[i]) != 1) {
            fprintf(stderr, "Error reading kernel value %d of %d\n", i, size);
            freeKernel(k);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    return k;
}

/*
 * Cada thread calcula un pixel RGB.
 * startRow i endRow defineixen quin tros de la imatge calcula aquesta GPU.
 * La imatge sencera esta a global memory; per tant, no cal comunicacio entre blocs.
 */
__global__ void convolveRGB_global_kernel(
        const int *__restrict__ inR,
        const int *__restrict__ inG,
        const int *__restrict__ inB,
        int *__restrict__ outR,
        int *__restrict__ outG,
        int *__restrict__ outB,
        int W,
        int H,
        int kx,
        int ky,
        int maxcolor,
        int startRow,
        int endRow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = startRow + blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= endRow || y >= H) return;

    int cx = kx / 2;
    int cy = ky / 2;

    float sumR = 0.0f;
    float sumG = 0.0f;
    float sumB = 0.0f;

    for (int m = 0; m < ky; m++) {
        int yy = y + m - cy;
        if (yy < 0 || yy >= H) continue;

        for (int n = 0; n < kx; n++) {
            int xx = x + n - cx;
            if (xx < 0 || xx >= W) continue;

            int imgIdx = yy * W + xx;

            /* Mateixa convencio que el codi OpenMP: kernel girat */
            float kval = d_kernel_const[(ky - 1 - m) * kx + (kx - 1 - n)];

            sumR += (float)inR[imgIdx] * kval;
            sumG += (float)inG[imgIdx] * kval;
            sumB += (float)inB[imgIdx] * kval;
        }
    }

    int outIdx = y * W + x;

    int r = (sumR >= 0.0f) ? (int)(sumR + 0.5f) : (int)(sumR - 0.5f);
    int g = (sumG >= 0.0f) ? (int)(sumG + 0.5f) : (int)(sumG - 0.5f);
    int b = (sumB >= 0.0f) ? (int)(sumB + 0.5f) : (int)(sumB - 0.5f);

    outR[outIdx] = clampDevice(r, maxcolor);
    outG[outIdx] = clampDevice(g, maxcolor);
    outB[outIdx] = clampDevice(b, maxcolor);
}

typedef struct {
    int deviceId;
    int startRow;
    int endRow;
    int *d_inR, *d_inG, *d_inB;
    int *d_outR, *d_outG, *d_outB;
    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
} GpuJob;

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr, "Usage: %s <image.ppm> <kernel.txt> <output.ppm> [num_gpus]\n", argv[0]);
        return EXIT_FAILURE;
    }

    double totalStart = nowSeconds();

    Imagen *img = readPPM(argv[1]);
    if (!img) {
        fprintf(stderr, "Error reading image\n");
        return EXIT_FAILURE;
    }

    Kernel *kern = readKernel(argv[2]);
    if (!kern) {
        fprintf(stderr, "Error reading kernel\n");
        freeImage(img);
        return EXIT_FAILURE;
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices found\n");
        freeImage(img);
        freeKernel(kern);
        return EXIT_FAILURE;
    }

    int requestedGpus = deviceCount;
    if (argc == 5) {
        requestedGpus = atoi(argv[4]);
        if (requestedGpus <= 0) requestedGpus = 1;
    }

    int numGpus = requestedGpus < deviceCount ? requestedGpus : deviceCount;
    if (numGpus > 2) {
        /* En el vostre servidor hi ha 2 RTX A4000. Es pot eliminar aquest limit si cal. */
        numGpus = 2;
    }

    int W = img->ancho;
    int H = img->altura;
    int maxcolor = img->maxcolor;
    long long numPixels = (long long)W * (long long)H;
    size_t bytes = (size_t)numPixels * sizeof(int);
    int kElems = kern->kx * kern->ky;

    Imagen *out = allocImage(W, H);
    if (!out) {
        fprintf(stderr, "Error allocating output image\n");
        freeImage(img);
        freeKernel(kern);
        return EXIT_FAILURE;
    }
    out->maxcolor = maxcolor;

    GpuJob *jobs = (GpuJob *)calloc(numGpus, sizeof(GpuJob));
    if (!jobs) {
        fprintf(stderr, "Error allocating jobs\n");
        freeImage(img);
        freeImage(out);
        freeKernel(kern);
        return EXIT_FAILURE;
    }

    printf("Image: %s\n", argv[1]);
    printf("Kernel: %s\n", argv[2]);
    printf("Output: %s\n", argv[3]);
    printf("Image size: %d x %d\n", W, H);
    printf("Kernel size: %d x %d\n", kern->kx, kern->ky);
    printf("CUDA devices used: %d\n", numGpus);
    printf("Block size: %d x %d = %d threads\n", BLOCK_X, BLOCK_Y, BLOCK_X * BLOCK_Y);
    printf("Memory strategy: global memory + L1/L2 cache, kernel in constant memory\n");

    /* Preparacio i llançament asíncron en cada GPU */
    for (int g = 0; g < numGpus; g++) {
        int startRow = (H * g) / numGpus;
        int endRow   = (H * (g + 1)) / numGpus;
        int rows     = endRow - startRow;

        jobs[g].deviceId = g;
        jobs[g].startRow = startRow;
        jobs[g].endRow = endRow;

        checkCuda(cudaSetDevice(g), "cudaSetDevice");

        cudaDeviceProp prop;
        checkCuda(cudaGetDeviceProperties(&prop, g), "cudaGetDeviceProperties");
        printf("GPU %d: %s | rows [%d, %d) | %d rows\n", g, prop.name, startRow, endRow, rows);

        checkCuda(cudaEventCreate(&jobs[g].startEvent), "cudaEventCreate startEvent");
        checkCuda(cudaEventCreate(&jobs[g].stopEvent), "cudaEventCreate stopEvent");

        checkCuda(cudaMalloc((void **)&jobs[g].d_inR, bytes), "cudaMalloc d_inR");
        checkCuda(cudaMalloc((void **)&jobs[g].d_inG, bytes), "cudaMalloc d_inG");
        checkCuda(cudaMalloc((void **)&jobs[g].d_inB, bytes), "cudaMalloc d_inB");
        checkCuda(cudaMalloc((void **)&jobs[g].d_outR, bytes), "cudaMalloc d_outR");
        checkCuda(cudaMalloc((void **)&jobs[g].d_outG, bytes), "cudaMalloc d_outG");
        checkCuda(cudaMalloc((void **)&jobs[g].d_outB, bytes), "cudaMalloc d_outB");

        checkCuda(cudaMemcpy(jobs[g].d_inR, img->R, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D R");
        checkCuda(cudaMemcpy(jobs[g].d_inG, img->G, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D G");
        checkCuda(cudaMemcpy(jobs[g].d_inB, img->B, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D B");

        checkCuda(cudaMemcpyToSymbol(d_kernel_const, kern->data, kElems * sizeof(float), 0, cudaMemcpyHostToDevice),
                  "cudaMemcpyToSymbol kernel");

        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
                  (rows + BLOCK_Y - 1) / BLOCK_Y);

        printf("GPU %d grid: %d x %d = %d blocks\n", g, grid.x, grid.y, grid.x * grid.y);

        checkCuda(cudaEventRecord(jobs[g].startEvent), "cudaEventRecord kernel start");
        convolveRGB_global_kernel<<<grid, block>>>(
            jobs[g].d_inR, jobs[g].d_inG, jobs[g].d_inB,
            jobs[g].d_outR, jobs[g].d_outG, jobs[g].d_outB,
            W, H, kern->kx, kern->ky, maxcolor, startRow, endRow
        );
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaEventRecord(jobs[g].stopEvent), "cudaEventRecord kernel stop");
    }

    float maxKernelMs = 0.0f;

    /* Sincronitzacio i copia parcial de resultats */
    for (int g = 0; g < numGpus; g++) {
        checkCuda(cudaSetDevice(jobs[g].deviceId), "cudaSetDevice sync");
        checkCuda(cudaEventSynchronize(jobs[g].stopEvent), "cudaEventSynchronize stopEvent");

        float kernelMs = 0.0f;
        checkCuda(cudaEventElapsedTime(&kernelMs, jobs[g].startEvent, jobs[g].stopEvent), "cudaEventElapsedTime");
        if (kernelMs > maxKernelMs) maxKernelMs = kernelMs;

        int startRow = jobs[g].startRow;
        int endRow = jobs[g].endRow;
        size_t offsetElems = (size_t)startRow * (size_t)W;
        size_t rowsElems = (size_t)(endRow - startRow) * (size_t)W;
        size_t rowsBytes = rowsElems * sizeof(int);

        checkCuda(cudaMemcpy(out->R + offsetElems, jobs[g].d_outR + offsetElems, rowsBytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H R rows");
        checkCuda(cudaMemcpy(out->G + offsetElems, jobs[g].d_outG + offsetElems, rowsBytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H G rows");
        checkCuda(cudaMemcpy(out->B + offsetElems, jobs[g].d_outB + offsetElems, rowsBytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy D2H B rows");
    }

    writePPM(argv[3], out);

    double totalStop = nowSeconds();
    double totalSeconds = totalStop - totalStart;

    printf("CUDA kernel time approx: %.6f seconds\n", maxKernelMs / 1000.0f);
    printf("Total time including IO and transfers: %.6f seconds\n", totalSeconds);

    for (int g = 0; g < numGpus; g++) {
        checkCuda(cudaSetDevice(jobs[g].deviceId), "cudaSetDevice free");
        cudaFree(jobs[g].d_inR);
        cudaFree(jobs[g].d_inG);
        cudaFree(jobs[g].d_inB);
        cudaFree(jobs[g].d_outR);
        cudaFree(jobs[g].d_outG);
        cudaFree(jobs[g].d_outB);
        cudaEventDestroy(jobs[g].startEvent);
        cudaEventDestroy(jobs[g].stopEvent);
    }


    free(jobs);
    freeImage(img);
    freeImage(out);
    freeKernel(kern);

    return EXIT_SUCCESS;
}
