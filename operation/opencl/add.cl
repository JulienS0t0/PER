__kernel void add(__global const float *x, __global float *y, const int n) {
    int index = get_global_id(0);
    int stride = get_global_size(0);

    for (int i = index; i < n; i += stride) {
        y[i] = x[i] + y[i];
    }
}
