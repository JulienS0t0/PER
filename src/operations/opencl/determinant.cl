// Kernel OpenCL pour le calcul du déterminant d'une matrice `float`
__kernel void determinantMatrixFloat(__global float *mat, __global float *result, int N) {
    int k = get_global_id(0); // Chaque work-item représente une ligne

    __local float tempMat[1024]; // Mémoire locale pour stocker la matrice temporaire

    // Chargement de la matrice dans la mémoire locale
    for (int j = 0; j < N; j++) {
        tempMat[k * N + j] = mat[k * N + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Élimination de Gauss pour former une matrice triangulaire
    for (int i = 0; i < N; i++) {
        if (k > i) {
            float facteur = tempMat[k * N + i] / tempMat[i * N + i];
            for (int j = i; j < N; j++) {
                tempMat[k * N + j] -= facteur * tempMat[i * N + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Calcul du produit des éléments diagonaux pour obtenir le déterminant
    if (k == 0) {
        float det = 1.0;
        for (int i = 0; i < N; i++) {
            det *= tempMat[i * N + i];
        }
        result[0] = det; // Stockage du résultat dans la mémoire globale
    }
}

// Kernel OpenCL pour le calcul du déterminant d'une matrice `int`
__kernel void determinantMatrixInt(__global int *mat, __global int *result, int N) {
    int k = get_global_id(0); // Chaque work-item représente une ligne

    __local int tempMat[1024]; // Mémoire locale pour stocker la matrice temporaire

    // Chargement de la matrice dans la mémoire locale
    for (int j = 0; j < N; j++) {
        tempMat[k * N + j] = mat[k * N + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Élimination de Gauss pour former une matrice triangulaire
    for (int i = 0; i < N; i++) {
        if (k > i) {
            int facteur = tempMat[k * N + i] / tempMat[i * N + i];
            for (int j = i; j < N; j++) {
                tempMat[k * N + j] -= facteur * tempMat[i * N + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Calcul du produit des éléments diagonaux pour obtenir le déterminant
    if (k == 0) {
        int det = 1;
        for (int i = 0; i < N; i++) {
            det *= tempMat[i * N + i];
        }
        result[0] = det; // Stockage du résultat dans la mémoire globale
    }
}
