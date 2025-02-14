#include <iostream>
#include <fstream>
#include <vector>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include "../../matrices/matrix_utils.h"

using namespace std;

const char *KERNEL_FILE = "./src/operations/opencl/transposition.cl";

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
        cerr << "Utilisation : " << argv[0] << " <fichier_matrice1.csv> [unused] [chemin_stockage]" << endl;
        return EXIT_FAILURE;
    }

    const char *fichier = argv[1];
    const char *chemin_stockage = (argc > 3) ? argv[3] : nullptr;

    bool is_float = type_matrice(fichier);
    int taille;
    void *h_mat = nullptr;
    charger_matrice_csv(fichier, &h_mat, &taille, is_float);

    int N = taille;
    int matrixSize = N * N * (is_float ? sizeof(float) : sizeof(int));
    void *h_result = malloc((is_float ? sizeof(float) : sizeof(int)) * matrixSize);

    // 1. Initialisation OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de la récupération de la plateforme OpenCL." << endl;
        return EXIT_FAILURE;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de la récupération de l'appareil OpenCL." << endl;
        return EXIT_FAILURE;
    }

    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de la création du contexte OpenCL." << endl;
        return EXIT_FAILURE;
    }

    cl_queue_properties properties[] = {0}; // Propriétés vides
    queue = clCreateCommandQueueWithProperties(context, device, properties, &err);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de la création de la file de commandes OpenCL." << endl;
        return EXIT_FAILURE;
    }

    // 2. Compilation du kernel
    string kernelSource = readKernelFile(KERNEL_FILE);
    const char *kernelSourceCStr = kernelSource.c_str();
    size_t kernelLength = kernelSource.length();

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourceCStr, &kernelLength, &err);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de la création du programme OpenCL." << endl;
        return EXIT_FAILURE;
    }

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        char *log = new char[logSize];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, nullptr);
        cerr << "Erreur lors de la compilation du kernel : " << endl << log << endl;
        delete[] log;
        return EXIT_FAILURE;
    }

    cl_kernel kernel = clCreateKernel(program, is_float ? "transpositionMatrixFloat" : "transpositionMatrixInt", &err);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de la création du kernel." << endl;
        return EXIT_FAILURE;
    }

    // 3. Allocation mémoire sur le GPU
    cl_mem d_mat = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize, h_mat, &err);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de l'allocation de mémoire pour la matrice d'entrée." << endl;
        return EXIT_FAILURE;
    }

    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (is_float ? sizeof(float) : sizeof(int))*matrixSize, nullptr, &err);
    // cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrixSize, nullptr, &err);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de l'allocation de mémoire pour la matrice résultat." << endl;
        return EXIT_FAILURE;
    }

    // 4. Passage des arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_mat);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_result);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &N);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors du passage des arguments au kernel." << endl;
        return EXIT_FAILURE;
    }

    // 5. Ajustement dynamique des tailles de travail
    size_t globalWorkSize[2] = {static_cast<size_t>(N), static_cast<size_t>(N)};
    size_t localWorkSize[2] = {16, 16};

    // Ajuste les tailles des groupes de travail si la matrice est plus petite que 16x16
    if (N < 16) {
        localWorkSize[0] = N;
        localWorkSize[1] = N;
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalWorkSize, localWorkSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de l'exécution du kernel." << endl;
        return EXIT_FAILURE;
    }

    // Lecture du résultat
    err = clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, (is_float ? sizeof(float) : sizeof(int)) * matrixSize, h_result, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur lors de la lecture du buffer." << endl;
        return EXIT_FAILURE;
    }

    // 6. Sauvegarde du résultat
    if (chemin_stockage) {
        sauvegarder_matrice_csv(chemin_stockage, h_result, N, is_float);
        cout << "Résultat enregistré dans le fichier : " << chemin_stockage << endl;
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
