#include <iostream>
#include <fstream>
#include <vector>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <ctime>
#include "../../matrices/matrix_utils.h"

using namespace std;

const char *KERNEL_FILE = "./src/operations/opencl/mul.cl";

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
        cerr << "Utilisation : " << argv[0] << " <fichier_matrice1.csv> <fichier_matrice2.csv> [chemin_stockage]" << endl;
        return EXIT_FAILURE;
    }

    const char *fichier1 = argv[1];
    const char *fichier2 = argv[2];
    const char *chemin_stockage = (argc > 3) ? argv[3] : nullptr;

    bool is_float = type_matrice(fichier1) || type_matrice(fichier2);
    int taille1, taille2;

    void *h_mat1 = nullptr, *h_mat2 = nullptr, *h_result = nullptr;
    charger_matrice_csv(fichier1, &h_mat1, &taille1, is_float);
    charger_matrice_csv(fichier2, &h_mat2, &taille2, is_float);

    if (taille1 != taille2) {
        cerr << "Erreur : Les matrices doivent avoir la même taille !" << endl;
        return EXIT_FAILURE;
    }

    int N = taille1;
    size_t matrixSize = N * N * (is_float ? sizeof(float) : sizeof(int));
    h_result = malloc(matrixSize);
    if (!h_result) {
        cerr << "Erreur d'allocation mémoire sur l'hôte." << endl;
        return EXIT_FAILURE;
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur : Impossible d'obtenir une plateforme OpenCL." << endl;
        return EXIT_FAILURE;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur : Aucun GPU OpenCL trouvé." << endl;
        return EXIT_FAILURE;
    }

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context) {
        cerr << "Erreur : Impossible de créer le contexte OpenCL." << endl;
        return EXIT_FAILURE;
    }

    queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    if (!queue) {
        cerr << "Erreur : Impossible de créer la file de commandes OpenCL." << endl;
        return EXIT_FAILURE;
    }

    string kernelSource = readKernelFile(KERNEL_FILE);
    const char *kernelSourceCStr = kernelSource.c_str();
    size_t kernelLength = kernelSource.length();

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCStr, &kernelLength, &err);
    if (!program) {
        cerr << "Erreur : Impossible de créer le programme OpenCL." << endl;
        return EXIT_FAILURE;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        cerr << "Erreur : Échec de la compilation du kernel OpenCL.\n" << buildLog.data() << endl;
        return EXIT_FAILURE;
    }

    cl_kernel kernel = clCreateKernel(program, is_float ? "mul_matrices_float" : "mul_matrices_int", &err);
    if (!kernel) {
        cerr << "Erreur : Impossible de créer le kernel OpenCL." << endl;
        return EXIT_FAILURE;
    }

    cl_mem d_mat1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize, h_mat1, &err);
    cl_mem d_mat2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize, h_mat2, &err);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrixSize, nullptr, &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_mat1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_mat2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_result);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    size_t globalWorkSize[2] = { (size_t)((N < 16) ? N : 16), 
                             (size_t)((N < 16) ? N : 16) };
    size_t localWorkSize[2] = { (size_t)((N < 16) ? N : 16), 
                            (size_t)((N < 16) ? N : 16) };

    
    clock_t start = clock();
    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de l'exécution du kernel OpenCL." << endl;
        return EXIT_FAILURE;
    }

    clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, matrixSize, h_result, 0, nullptr, nullptr);
    clock_t end = clock();
    
    printf("Multiplication terminée en %.2f ms.\n", ((double)(end - start)) / CLOCKS_PER_SEC * 1000);

    if (chemin_stockage) {
        sauvegarder_matrice_csv(chemin_stockage, h_result, N, is_float);
        cout << "Résultat enregistré dans le fichier : " << chemin_stockage << endl;
        cout << "Premiers éléments du résultat : " << ((is_float) ? ((float*)h_result)[0] : ((int*)h_result)[0]) << endl;

    }

    clReleaseMemObject(d_mat1);
    clReleaseMemObject(d_mat2);
    clReleaseMemObject(d_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_result);
    return EXIT_SUCCESS;
}
