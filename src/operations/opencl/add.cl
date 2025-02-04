__kernel void add_matrices_float(__global float *mat1, __global float *mat2, __global float *result, int N) {
    int index = get_global_id(0);
    if (index < N * N) {
        result[index] = mat1[index] + mat2[index];
    }
}

__kernel void add_matrices_int(__global int *mat1, __global int *mat2, __global int *result, int N) {
    int index = get_global_id(0);
    if (index < N * N) {
        result[index] = mat1[index] + mat2[index];
    }
}
