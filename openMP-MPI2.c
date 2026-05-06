#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

/* ================== ESTRUCTURES ================== */

typedef struct {
    int ancho, altura, maxcolor;
    int *R, *G, *B;
} Imagen;

typedef struct {
    int kx, ky;
    float *data;
} Kernel;

/* ================== UTILITATS ================== */

Imagen *allocImage(int w, int h) {
    Imagen *img = malloc(sizeof(Imagen));
    if (!img) return NULL;

    int size = w * h;

    img->ancho = w;
    img->altura = h;
    img->maxcolor = 255;

    img->R = malloc(size * sizeof(int));
    img->G = malloc(size * sizeof(int));
    img->B = malloc(size * sizeof(int));

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

/* ================== LECTURA PPM ================== */

Imagen *readPPM(char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Error opening image");
        return NULL;
    }

    char format[3];
    fscanf(fp, "%2s", format);

    if (strcmp(format, "P3") != 0) {
        printf("Only P3 format supported\n");
        fclose(fp);
        return NULL;
    }

    int c = fgetc(fp);
    while (c == '\n' || c == ' ' || c == '\t') {
        c = fgetc(fp);
    }

    while (c == '#') {
        while (c != '\n' && c != EOF) {
            c = fgetc(fp);
        }
        c = fgetc(fp);
    }

    ungetc(c, fp);

    int w, h, max;
    fscanf(fp, "%d %d", &w, &h);
    fscanf(fp, "%d", &max);

    Imagen *img = allocImage(w, h);
    if (!img) {
        fclose(fp);
        return NULL;
    }

    img->maxcolor = max;

    int size = w * h;

    for (int i = 0; i < size; i++) {
        fscanf(fp, "%d %d %d", &img->R[i], &img->G[i], &img->B[i]);
    }

    fclose(fp);
    return img;
}

/* ================== ESCRIPTURA PPM ================== */

void writePPM(char *filename, Imagen *img) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        perror("Error writing image");
        return;
    }

    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", img->ancho, img->altura);
    fprintf(fp, "%d\n", img->maxcolor);

    int size = img->ancho * img->altura;

    for (int i = 0; i < size; i++) {
        fprintf(fp, "%d %d %d ", img->R[i], img->G[i], img->B[i]);

        if ((i + 1) % img->ancho == 0) {
            fprintf(fp, "\n");
        }
    }

    fclose(fp);
}

/* ================== LECTURA KERNEL ================== */

Kernel *readKernel(char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Kernel error");
        return NULL;
    }

    Kernel *k = malloc(sizeof(Kernel));
    if (!k) {
        fclose(fp);
        return NULL;
    }

    fscanf(fp, "%d,%d,", &k->kx, &k->ky);

    int size = k->kx * k->ky;
    k->data = malloc(size * sizeof(float));

    if (!k->data) {
        free(k);
        fclose(fp);
        return NULL;
    }

    for (int i = 0; i < size - 1; i++) {
        fscanf(fp, "%f,", &k->data[i]);
    }

    fscanf(fp, "%f", &k->data[size - 1]);

    fclose(fp);
    return k;
}

/* ================== CONVOLUCIÓ ================== */

int clamp(int v, int max) {
    if (v < 0) return 0;
    if (v > max) return max;
    return v;
}

void convolve2D_omp(int *in, int *out,
                    int W, int H,
                    float *kernel,
                    int kx, int ky,
                    int maxcolor) {

    int cx = kx / 2;
    int cy = ky / 2;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {

            float sum = 0.0f;

            for (int m = 0; m < ky; m++) {
                for (int n = 0; n < kx; n++) {

                    int ii = i + m - cy;
                    int jj = j + n - cx;

                    if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                        sum += in[ii * W + jj] *
                               kernel[(ky - 1 - m) * kx + (kx - 1 - n)];
                    }
                }
            }

            if (sum >= 0.0f)
                out[i * W + j] = clamp((int)(sum + 0.5f), maxcolor);
            else
                out[i * W + j] = clamp((int)(sum - 0.5f), maxcolor);
        }
    }
}

/* ================== MAIN ================== */

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank, nprocs;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (nprocs != 4) {
        if (rank == 0) {
            printf("Error: aquest programa s'ha d'executar amb 4 processos MPI.\n");
            printf("Ús: mpirun -np 4 ./mpi_rgb_conv image.ppm kernel.txt output.ppm\n");
        }

        MPI_Finalize();
        return -1;
    }

    if (argc != 4) {
        if (rank == 0) {
            printf("Usage: %s <image.ppm> <kernel.txt> <output.ppm>\n", argv[0]);
        }

        MPI_Finalize();
        return -1;
    }

    Imagen *img = NULL;
    Kernel *kern = NULL;

    int W = 0, H = 0, maxcolor = 0;
    int kx = 0, ky = 0, ksize = 0;
    int sizeImg = 0;

    double start_total = 0.0;
    double elapsed_total = 0.0;

    /* ---------- RANK 0: LECTURA ---------- */

    if (rank == 0) {
        start_total = MPI_Wtime();

        img = readPPM(argv[1]);
        if (!img) {
            fprintf(stderr, "Error reading image\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        kern = readKernel(argv[2]);
        if (!kern) {
            fprintf(stderr, "Error reading kernel\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        W = img->ancho;
        H = img->altura;
        maxcolor = img->maxcolor;

        kx = kern->kx;
        ky = kern->ky;
        ksize = kx * ky;
    }

    /* ---------- BROADCAST DE DADES GLOBALS ---------- */

    MPI_Bcast(&W, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&H, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxcolor, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kx, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ky, 1, MPI_INT, 0, MPI_COMM_WORLD);

    ksize = kx * ky;
    sizeImg = W * H;

    float *kernel = malloc(ksize * sizeof(float));
    if (!kernel) {
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    if (rank == 0) {
        memcpy(kernel, kern->data, ksize * sizeof(float));
    }

    MPI_Bcast(kernel, ksize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* ---------- MASTER-WORKER RGB ---------- */

    if (rank == 0) {

        MPI_Send(img->R, sizeImg, MPI_INT, 1, 0, MPI_COMM_WORLD);
        MPI_Send(img->G, sizeImg, MPI_INT, 2, 0, MPI_COMM_WORLD);
        MPI_Send(img->B, sizeImg, MPI_INT, 3, 0, MPI_COMM_WORLD);

        MPI_Recv(img->R, sizeImg, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(img->G, sizeImg, MPI_INT, 2, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(img->B, sizeImg, MPI_INT, 3, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        writePPM(argv[3], img);

        elapsed_total = MPI_Wtime() - start_total;

        printf("Image: %s\n", argv[1]);
        printf("Kernel: %s\n", argv[2]);
        printf("Output: %s\n", argv[3]);
        printf("Image size: %d x %d\n", W, H);
        printf("Kernel size: %d x %d\n", kx, ky);
        printf("MPI processes: %d\n", nprocs);
        printf("OpenMP threads per worker: %d\n", omp_get_max_threads());
        printf("Total execution time rank 0: %.6f seconds\n", elapsed_total);

    } else {

        int *in = malloc(sizeImg * sizeof(int));
        int *out = malloc(sizeImg * sizeof(int));

        if (!in || !out) {
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        MPI_Recv(in, sizeImg, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        convolve2D_omp(in, out, W, H, kernel, kx, ky, maxcolor);

        MPI_Send(out, sizeImg, MPI_INT, 0, 1, MPI_COMM_WORLD);

        free(in);
        free(out);
    }

    free(kernel);
ls

    if (rank == 0) {
        freeImage(img);
        freeKernel(kern);
    }

    MPI_Finalize();
    return 0;
}