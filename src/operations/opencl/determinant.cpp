#include <iostream>
#include <fstream>
#include <vector>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include "../../matrices/matrix_utils.h"

using namespace std;

const char *KERNEL_FILE = "./src/operations/opencl/determinant.cl";

// Fonction pour lire un fichier de kernel OpenCL
string readKernelFile(const char *filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Erreur : Impossible de lire le fichier du kernel OpenCL : " << filename << endl;
        exit(EXIT_FAILURE);
    }
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Utilisation : " << argv[0] << " <fichier_matrice.csv>" << endl;
        return EXIT_FAILURE;
    }

    const char *fichier = argv[1];

    bool is_float = type_matrice(fichier);
    int taille;
    void *h_mat = nullptr;
    charger_matrice_csv(fichier, &h_mat, &taille, is_float);

    int N = taille;
    int matrixSize = N * N * (is_float ? sizeof(float) : sizeof(int));
    void *h_result = malloc(is_float ? sizeof(float) : sizeof(int));

    // 1. Initialisation OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    err = clGetPlatformIDs(1, &platform, nullptr);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_queue_properties properties[] = {0}; // Propriétés vides
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);

    // 2. Compilation du kernel
    string kernelSource = readKernelFile(KERNEL_FILE);
    const char *kernelSourceCStr = kernelSource.c_str();
    size_t kernelLength = kernelSource.length();

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCStr, &kernelLength, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    cl_kernel kernel = clCreateKernel(program, is_float ? "determinantMatrixFloat" : "determinantMatrixInt", &err);

    // 3. Allocation mémoire sur le GPU
    cl_mem d_mat = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize, h_mat, &err);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, is_float ? sizeof(float) : sizeof(int), nullptr, &err);

    // 4. Passage des arguments
    clock_t start = clock();
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_mat);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_result);
    clSetKernelArg(kernel, 2, sizeof(int), &N);

    // 5. Exécution du kernel
    size_t globalWorkSize = N;
    size_t localWorkSize = (globalWorkSize < 256) ? globalWorkSize : 256;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    clock_t end = clock();

    err = clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, is_float ? sizeof(float) : sizeof(int), h_result, 0, nullptr, nullptr);

    double temps_execution = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Déterminant terminé en %.2f ms.\n", temps_execution);

    if (is_float) {
    printf("Déterminant : %.6f\n", *(float*)h_result);
    } else {
        printf("Déterminant : %d\n", *(int*)h_result);
    }

    // Nettoyage
    clReleaseMemObject(d_mat);
    clReleaseMemObject(d_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(h_result);
    free(h_mat);

    return EXIT_SUCCESS;
}
