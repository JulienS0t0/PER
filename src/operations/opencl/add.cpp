#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

// Helper function to load the kernel source code
std::string loadKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open kernel file.");
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

int main() {
    const int N = 1 << 20; // 1 million elements
    std::vector<float> x(N, 1.0f); // Initialize x with 1.0
    std::vector<float> y(N, 2.0f); // Initialize y with 2.0

    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem d_x, d_y;

    // Get platform and device
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    // Create context and command queue
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueueWithProperties(context, device, nullptr, nullptr);

    // Create buffers
    d_x = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), x.data(), nullptr);
    d_y = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(float), y.data(), nullptr);

    // Load and build kernel
    std::string kernelSource = loadKernelSource("add.cl");
    const char* source = kernelSource.c_str();
    size_t sourceSize = kernelSource.size();
    program = clCreateProgramWithSource(context, 1, &source, &sourceSize, nullptr);
    clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    // Create kernel
    kernel = clCreateKernel(program, "add", nullptr);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_x);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_y);
    clSetKernelArg(kernel, 2, sizeof(int), &N);

    // Launch kernel
    size_t globalWorkSize = 256;  // Number of threads
    size_t localWorkSize = 64;    // Threads per block
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);

    // Copy result back to host
    clEnqueueReadBuffer(queue, d_y, CL_TRUE, 0, N * sizeof(float), y.data(), 0, nullptr, nullptr);

    // Verify results
    for (int i = 0; i < 10; i++) {
        std::cout << y[i] << std::endl;
    }

    // Clean up
    clReleaseMemObject(d_x);
    clReleaseMemObject(d_y);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
