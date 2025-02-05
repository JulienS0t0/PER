__kernel void traceMatrixFloat(__global float *mat, __global float *result, int N) {
    int gid = get_global_id(0);
    if (gid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += mat[i * (N + 1)];
        }
        result[0] = sum;
    }
}

__kernel void traceMatrixInt(__global int *mat, __global int *result, int N) {
    int gid = get_global_id(0);
    if (gid == 0) {
        int sum = 0;
        for (int i = 0; i < N; i++) {
            sum += mat[i * (N + 1)];
        }
        result[0] = sum;
    }
}