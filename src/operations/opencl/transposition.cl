// Kernel OpenCL pour la transposition d'une matrice `float`
__kernel void transpositionMatrixFloat(__global float *mat, __global float *result, int N) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < N && col < N) {
        result[col * N + row] = mat[row * N + col]; // Transposition de l'élément
    }
}

// Kernel OpenCL pour la transposition d'une matrice `int`
__kernel void transpositionMatrixInt(__global int *mat, __global int *result, int N) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < N && col < N) {
        result[col * N + row] = mat[row * N + col]; // Transposition de l'élément
    }
}
