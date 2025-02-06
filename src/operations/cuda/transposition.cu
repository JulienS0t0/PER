#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include "../../matrices/matrix_utils.h"

using namespace std;
using namespace std::chrono;

// CUDA Kernel pour la transposition d'une matrice `float`
__global__
void transpositionMatrixFloat(float *mat, float *result, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        result[col * N + row] = mat[row * N + col];
    }
}

// CUDA Kernel pour la transposition d'une matrice `int`
__global__
void transpositionMatrixInt(int *mat, int *result, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        result[col * N + row] = mat[row * N + col];
    }
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Utilisation : " << argv[0] << " <fichier_matrice1.csv>" << endl;
        return EXIT_FAILURE;
    }

    const char *fichier1 = argv[1];
    bool is_float = type_matrice(fichier1);
    int taille;
    void *h_mat = nullptr, *h_result = nullptr;
    void *d_mat = nullptr, *d_result = nullptr;

    // Charger la matrice
    charger_matrice_csv(fichier1, &h_mat, &taille, is_float);

    h_result = malloc(is_float ? sizeof(float) : sizeof(int));
    if (!h_result) {
        cerr << "Erreur d'allocation mémoire" << endl;
        free(h_mat);
        return EXIT_FAILURE;
    }

    auto start = high_resolution_clock::now();
    cudaMalloc(&d_mat, taille * taille * (is_float ? sizeof(float) : sizeof(int)));
    cudaMalloc(&d_result, is_float ? sizeof(float) : sizeof(int));
    cudaMemcpy(d_mat, h_mat, taille * taille * (is_float ? sizeof(float) : sizeof(int)), cudaMemcpyHostToDevice);

    int blockSize = 256;
    if (is_float) {
        transpositionMatrixFloat<<<1, blockSize>>>((float*)d_mat, (float*)d_result, taille);
    } else {
        transpositionMatrixInt<<<1, blockSize>>>((int*)d_mat, (int*)d_result, taille);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(h_result, d_result, is_float ? sizeof(float) : sizeof(int), cudaMemcpyDeviceToHost);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    // cout << "Trace terminée en " << duration.count() << " ms sur GPU (CUDA)." << endl;
    // cout << "Résultat de la trace : " << (is_float ? *(float*)h_result : *(int*)h_result) << endl;

    free(h_mat);
    free(h_result);
    cudaFree(d_mat);
    cudaFree(d_result);

    return EXIT_SUCCESS;
}
