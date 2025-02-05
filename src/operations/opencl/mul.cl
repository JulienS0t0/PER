// OpenCL Kernel pour la multiplication des matrices `float`
__kernel void mul_matrices_float(__global float *mat1, __global float *mat2, __global float *result, int N) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < N && col < N) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += mat1[row * N + k] * mat2[k * N + col];
        }
        result[row * N + col] = sum;
    }
}

// OpenCL Kernel pour la multiplication des matrices `int`
__kernel void mul_matrices_int(__global int *mat1, __global int *mat2, __global int *result, int N) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += mat1[row * N + k] * mat2[k * N + col];
        }
        result[row * N + col] = sum;
    }
}
