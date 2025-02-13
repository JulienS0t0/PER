#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include "../../matrices/matrix_utils.h"

using namespace std;
using namespace std::chrono;

// CUDA Kernel pour la multiplication des matrices `float`
__global__
void multiplyMatricesFloat(float *mat1, float *mat2, float *result, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += mat1[row * N + k] * mat2[k * N + col];
        }
        result[row * N + col] = sum;
    }
}

// CUDA Kernel pour la multiplication des matrices `int`
__global__
void multiplyMatricesInt(int *mat1, int *mat2, int *result, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += mat1[row * N + k] * mat2[k * N + col];
        }
        result[row * N + col] = sum;
    }
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Utilisation : " << argv[0] << " <fichier_matrice1.csv> <fichier_matrice2.csv>" << endl;
        return EXIT_FAILURE;
    }

    const char *fichier1 = argv[1];
    const char *fichier2 = argv[2];

    bool is_float = type_matrice(fichier1) || type_matrice(fichier2);
    int taille1, taille2;

    void *h_mat1 = nullptr, *h_mat2 = nullptr, *h_result = nullptr;
    void *d_mat1 = nullptr, *d_mat2 = nullptr, *d_result = nullptr;

    // Charger les matrices
    charger_matrice_csv(fichier1, &h_mat1, &taille1, is_float);
    charger_matrice_csv(fichier2, &h_mat2, &taille2, is_float);

    if (taille1 != taille2) {
        cerr << "Erreur : Les matrices doivent avoir la même taille." << endl;
        free(h_mat1);
        free(h_mat2);
        return EXIT_FAILURE;
    }

    int N = taille1;
    int matrixSize = N * N * (is_float ? sizeof(float) : sizeof(int));

    int blockSize = 256;
    int numBlocks = (N * N + blockSize - 1) / blockSize;

    h_result = malloc(matrixSize);
    if (!h_result) {
        cerr << "Erreur d'allocation mémoire sur l'hôte." << endl;
        free(h_mat1);
        free(h_mat2);
        return EXIT_FAILURE;
    }

    auto start = high_resolution_clock::now();

    cudaMalloc(&d_mat1, matrixSize);
    cudaMalloc(&d_mat2, matrixSize);
    cudaMalloc(&d_result, matrixSize);

    cudaMemcpy(d_mat1, h_mat1, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat2, h_mat2, matrixSize, cudaMemcpyHostToDevice);

    if (is_float) {
        multiplyMatricesFloat<<<numBlocks, blockSize>>>((float*)d_mat1, (float*)d_mat2, (float*)d_result, N);
    } else {
        multiplyMatricesInt<<<numBlocks, blockSize>>>((int*)d_mat1, (int*)d_mat2, (int*)d_result, N);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, d_result, matrixSize, cudaMemcpyDeviceToHost);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Multiplication terminée en " << duration.count() << " ms sur GPU (CUDA)." << endl;

    // char nom_fichier[256];
    // generer_nom_fichier_resultat(nom_fichier, sizeof(nom_fichier), "res/cuda", "mul", is_float, N);
    // sauvegarder_matrice_csv(nom_fichier, h_result, N, is_float);
    // cout << "Résultat enregistré dans : " << nom_fichier << endl;

    free(h_mat1);
    free(h_mat2);
    free(h_result);
    cudaFree(d_mat1);
    cudaFree(d_mat2);
    cudaFree(d_result);


    return EXIT_SUCCESS;
}
