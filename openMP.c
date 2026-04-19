#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>

// Structure to store image contents
struct imagenppm{
    int altura;
    int ancho;
    char *comentario;
    int maxcolor;
    int P;
    int *R;
    int *G;
    int *B;
};
typedef struct imagenppm* ImagenData;

// Structure to store kernel contents
struct structkernel{
    int kernelX;
    int kernelY;
    float *vkern;
};
typedef struct structkernel* kernelData;

// Function declarations
ImagenData initimage(char* nombre, FILE **fp, int partitions, int halo);
ImagenData duplicateImageData(ImagenData src, int partitions, int halo);

int readImage(ImagenData Img, FILE **fp, int dim, int halosize, long int *position);
int duplicateImageChunk(ImagenData src, ImagenData dst, int dim);
int initfilestore(ImagenData img, FILE **fp, char* nombre, long *position);
int savingChunk(ImagenData img, FILE **fp, int dim, int offset);
int convolve2D_omp(int* in, int* out, int dataSizeX, int dataSizeY,
                   float* kernel, int kernelSizeX, int kernelSizeY);
void freeImagestructure(ImagenData *src);
kernelData leerKernel(char* nombre);

// Open image file and initialize image structure
ImagenData initimage(char* nombre, FILE **fp, int partitions, int halo){
    char c;
    char comentario[300];
    int i = 0, chunk = 0;
    ImagenData img = NULL;

    if ((*fp = fopen(nombre, "r")) == NULL){
        perror("Error opening image");
    }
    else{
        img = (ImagenData) malloc(sizeof(struct imagenppm));
        if (img == NULL) return NULL;

        fscanf(*fp, "%c%d ", &c, &(img->P));

        while((c = fgetc(*fp)) != '\n'){
            comentario[i] = c;
            i++;
        }
        comentario[i] = '\0';

        img->comentario = calloc(strlen(comentario) + 1, sizeof(char));
        if (img->comentario == NULL) return NULL;
        strcpy(img->comentario, comentario);

        fscanf(*fp, "%d %d %d", &img->ancho, &img->altura, &img->maxcolor);

        chunk = img->ancho * img->altura / partitions;
        chunk = chunk + img->ancho * halo;

        if ((img->R = calloc(chunk, sizeof(int))) == NULL) return NULL;
        if ((img->G = calloc(chunk, sizeof(int))) == NULL) return NULL;
        if ((img->B = calloc(chunk, sizeof(int))) == NULL) return NULL;
    }
    return img;
}

// Duplicate image structure for output image
ImagenData duplicateImageData(ImagenData src, int partitions, int halo){
    int chunk = 0;
    ImagenData dst = (ImagenData) malloc(sizeof(struct imagenppm));
    if (dst == NULL) return NULL;

    dst->P = src->P;

    dst->comentario = calloc(strlen(src->comentario) + 1, sizeof(char));
    if (dst->comentario == NULL) return NULL;
    strcpy(dst->comentario, src->comentario);

    dst->ancho = src->ancho;
    dst->altura = src->altura;
    dst->maxcolor = src->maxcolor;

    chunk = dst->ancho * dst->altura / partitions;
    chunk = chunk + src->ancho * halo;

    if ((dst->R = calloc(chunk, sizeof(int))) == NULL) return NULL;
    if ((dst->G = calloc(chunk, sizeof(int))) == NULL) return NULL;
    if ((dst->B = calloc(chunk, sizeof(int))) == NULL) return NULL;

    return dst;
}

// Read image chunk
int readImage(ImagenData img, FILE **fp, int dim, int halosize, long *position){
    int i = 0, haloposition = 0;

    if (fseek(*fp, *position, SEEK_SET))
        perror("Error in fseek");

    haloposition = dim - (img->ancho * halosize * 2);

    for(i = 0; i < dim; i++) {
        if (halosize != 0 && i == haloposition) *position = ftell(*fp);
        fscanf(*fp, "%d %d %d ", &img->R[i], &img->G[i], &img->B[i]);
    }

    return 0;
}

// Copy image chunk
int duplicateImageChunk(ImagenData src, ImagenData dst, int dim){
    int i;
    for(i = 0; i < dim; i++){
        dst->R[i] = src->R[i];
        dst->G[i] = src->G[i];
        dst->B[i] = src->B[i];
    }
    return 0;
}

// Read kernel
kernelData leerKernel(char* nombre){
    FILE *fp;
    int i = 0;
    kernelData kern = NULL;

    fp = fopen(nombre, "r");
    if(!fp){
        perror("Error opening kernel");
    }
    else{
        kern = (kernelData) malloc(sizeof(struct structkernel));
        if (kern == NULL) return NULL;

        fscanf(fp, "%d,%d,", &kern->kernelX, &kern->kernelY);
        kern->vkern = (float *)malloc(kern->kernelX * kern->kernelY * sizeof(float));
        if (kern->vkern == NULL) return NULL;

        for (i = 0; i < (kern->kernelX * kern->kernelY) - 1; i++){
            fscanf(fp, "%f,", &kern->vkern[i]);
        }
        fscanf(fp, "%f", &kern->vkern[i]);
        fclose(fp);
    }
    return kern;
}

// Initialize output file
int initfilestore(ImagenData img, FILE **fp, char* nombre, long *position){
    if ((*fp = fopen(nombre, "w")) == NULL){
        perror("Error creating output file");
        return -1;
    }

    fprintf(*fp, "P%d\n%s\n%d %d\n%d\n",
            img->P, img->comentario, img->ancho, img->altura, img->maxcolor);

    *position = ftell(*fp);
    return 0;
}

// Save chunk
int savingChunk(ImagenData img, FILE **fp, int dim, int offset){
    int i;
    for(i = offset; i < dim + offset; i++){
        fprintf(*fp, "%d %d %d ", img->R[i], img->G[i], img->B[i]);
    }
    return 0;
}

// Free image structure
void freeImagestructure(ImagenData *src){
    if (*src == NULL) return;

    free((*src)->comentario);
    free((*src)->R);
    free((*src)->G);
    free((*src)->B);
    free(*src);
}

// Thread-safe OpenMP convolution
int convolve2D_omp(int* in, int* out, int dataSizeX, int dataSizeY,
                   float* kernel, int kernelSizeX, int kernelSizeY)
{
    if(!in || !out || !kernel) return -1;
    if(dataSizeX <= 0 || dataSizeY <= 0 || kernelSizeX <= 0 || kernelSizeY <= 0) return -1;

    int kCenterX = kernelSizeX / 2;
    int kCenterY = kernelSizeY / 2;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < dataSizeY; i++) {
        for (int j = 0; j < dataSizeX; j++) {
            float sum = 0.0f;

            for (int m = 0; m < kernelSizeY; m++) {
                int mm = kernelSizeY - 1 - m;   // flipped kernel row
                int rowIndex = i + (m - kCenterY);

                if (rowIndex < 0 || rowIndex >= dataSizeY)
                    continue;

                for (int n = 0; n < kernelSizeX; n++) {
                    int nn = kernelSizeX - 1 - n; // flipped kernel col
                    int colIndex = j + (n - kCenterX);

                    if (colIndex < 0 || colIndex >= dataSizeX)
                        continue;

                    sum += in[rowIndex * dataSizeX + colIndex] *
                           kernel[mm * kernelSizeX + nn];
                }
            }

            if (sum >= 0.0f)
                out[i * dataSizeX + j] = (int)(sum + 0.5f);
            else
                out[i * dataSizeX + j] = (int)(sum - 0.5f);
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        printf("Usage: %s <image-file> <kernel-file> <result-file> <partitions>\n", argv[0]);
        printf("\nError, missing parameters:\n");
        printf("format: ./ompconv image_file kernel_file result_file partitions\n");
        printf("- image_file : source image path (*.ppm)\n");
        printf("- kernel_file: kernel path (text file with 1D kernel matrix)\n");
        printf("- result_file: result image path (*.ppm)\n");
        printf("- partitions : image partitions\n\n");
        return -1;
    }

    int imagesize, partitions, partsize, chunksize, halo, halosize;
    long position = 0;

    double tstart = 0.0, tend = 0.0;
    double tread = 0.0, tcopy = 0.0, tconv = 0.0, tstore = 0.0, treadk = 0.0;
    double start = 0.0;

    FILE *fpsrc = NULL, *fpdst = NULL;
    ImagenData source = NULL, output = NULL;
    kernelData kern = NULL;

    partitions = atoi(argv[4]);

    tstart = omp_get_wtime();

    // Read kernel
    start = omp_get_wtime();
    kern = leerKernel(argv[2]);
    if (kern == NULL) return -1;

    if (partitions == 1) halo = 0;
    else halo = (kern->kernelY / 2) * 2;

    treadk += omp_get_wtime() - start;

    // Read image header and allocate memory
    start = omp_get_wtime();
    source = initimage(argv[1], &fpsrc, partitions, halo);
    if (source == NULL) return -1;
    tread += omp_get_wtime() - start;

    // Duplicate image structure
    start = omp_get_wtime();
    output = duplicateImageData(source, partitions, halo);
    if (output == NULL) return -1;
    tcopy += omp_get_wtime() - start;

    // Initialize output file
    start = omp_get_wtime();
    if (initfilestore(output, &fpdst, argv[3], &position) != 0) {
        return -1;
    }
    tstore += omp_get_wtime() - start;

    // Process chunks
    int c = 0, offset = 0;
    imagesize = source->altura * source->ancho;
    partsize  = imagesize / partitions;

    while (c < partitions) {
        // Read next chunk
        start = omp_get_wtime();

        if (c == 0) {
            halosize  = halo / 2;
            chunksize = partsize + (source->ancho * halosize);
            offset    = 0;
        }
        else if (c < partitions - 1) {
            halosize  = halo;
            chunksize = partsize + (source->ancho * halosize);
            offset    = (source->ancho * halo / 2);
        }
        else {
            halosize  = halo / 2;
            chunksize = partsize + (source->ancho * halosize);
            offset    = (source->ancho * halo / 2);
        }

        if (readImage(source, &fpsrc, chunksize, halo / 2, &position)) {
            return -1;
        }
        tread += omp_get_wtime() - start;

        // Copy chunk
        start = omp_get_wtime();
        if (duplicateImageChunk(source, output, chunksize)) {
            return -1;
        }
        tcopy += omp_get_wtime() - start;

        // Convolution
        start = omp_get_wtime();

        int localHeight = (source->altura / partitions) + halosize;

        convolve2D_omp(source->R, output->R, source->ancho, localHeight,
                       kern->vkern, kern->kernelX, kern->kernelY);

        convolve2D_omp(source->G, output->G, source->ancho, localHeight,
                       kern->vkern, kern->kernelX, kern->kernelY);

        convolve2D_omp(source->B, output->B, source->ancho, localHeight,
                       kern->vkern, kern->kernelX, kern->kernelY);

        tconv += omp_get_wtime() - start;

        // Save chunk
        start = omp_get_wtime();
        if (savingChunk(output, &fpdst, partsize, offset)) {
            return -1;
        }
        tstore += omp_get_wtime() - start;

        c++;
    }

    fclose(fpsrc);
    fclose(fpdst);

    tend = omp_get_wtime();

    printf("Image: %s\n", argv[1]);
    printf("ISizeX : %d\n", source->ancho);
    printf("ISizeY : %d\n", source->altura);
    printf("kSizeX : %d\n", kern->kernelX);
    printf("kSizeY : %d\n", kern->kernelY);
    printf("Using %d OpenMP threads\n", omp_get_max_threads());
    printf("%.6lf seconds elapsed for reading image file.\n", tread);
    printf("%.6lf seconds elapsed for copying image structure.\n", tcopy);
    printf("%.6lf seconds elapsed for reading kernel matrix.\n", treadk);
    printf("%.6lf seconds elapsed for making the convolution.\n", tconv);
    printf("%.6lf seconds elapsed for writing the resulting image.\n", tstore);
    printf("%.6lf seconds elapsed total.\n", tend - tstart);

    free(kern->vkern);
    free(kern);
    freeImagestructure(&source);
    freeImagestructure(&output);

    return 0;
}