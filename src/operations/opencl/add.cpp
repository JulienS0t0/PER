#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl.h>  // OpenCL C API
#include <unistd.h>
#include "../../matrices/matrix_utils.h"

using namespace std;

const char *KERNEL_FILE = "./src/operations/opencl/add.cl";

string readKernelFile(const char *filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Erreur : Impossible de lire le fichier du kernel OpenCL : " << filename << endl;
        exit(EXIT_FAILURE);
    }
    return string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Utilisation : " << argv[0] << " <fichier_matrice1.csv> <fichier_matrice2.csv>" << endl;
        return EXIT_FAILURE;
    }

    const char *fichier1 = argv[1];
    const char *fichier2 = argv[2];

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
    int matrixSize = N * N * (is_float ? sizeof(float) : sizeof(int));
    h_result = malloc(matrixSize);  // Allocate memory for the result matrix

    // 1. Initialize OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    
    // Get the first available platform
    err = clGetPlatformIDs(1, &platform, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur : Impossible d'obtenir une plateforme OpenCL." << endl;
        return EXIT_FAILURE;
    }

    // Get the first available GPU device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur : Aucun GPU OpenCL trouvé." << endl;
        return EXIT_FAILURE;
    }

    // Create an OpenCL context
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (!context) {
        cerr << "Erreur : Impossible de créer le contexte OpenCL." << endl;
        return EXIT_FAILURE;
    }

    // Create a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    if (!queue) {
        cerr << "Erreur : Impossible de créer la file de commandes OpenCL." << endl;
        return EXIT_FAILURE;
    }

    // 2. Load & Compile OpenCL Kernel
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
        cerr << "Erreur : Échec de la compilation du kernel OpenCL." << endl;
        
        // Print build log for debugging
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        vector<char> buildLog(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
        cerr << "Build Log:\n" << buildLog.data() << endl;

        return EXIT_FAILURE;
    }

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, is_float ? "add_matrices_float" : "add_matrices_int", &err);
    if (!kernel) {
        cerr << "Erreur : Impossible de créer le kernel OpenCL." << endl;
        return EXIT_FAILURE;
    }

    // 3. Allocate Buffers on Device
    cl_mem d_mat1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize, h_mat1, &err);
    cl_mem d_mat2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize, h_mat2, &err);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrixSize, nullptr, &err);

    if (!d_mat1 || !d_mat2 || !d_result) {
        cerr << "Erreur : Échec de l'allocation de la mémoire sur le GPU." << endl;
        return EXIT_FAILURE;
    }

    // 4. Set Kernel Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_mat1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_mat2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_result);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    // 5. Execute Kernel
    size_t globalWorkSize = N * N;
    size_t localWorkSize = (globalWorkSize < 256) ? globalWorkSize : 256;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur : Échec de l'exécution du kernel OpenCL." << endl;
        return EXIT_FAILURE;
    }

    // 6. Read Back Result
    err = clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, matrixSize, h_result, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        cerr << "Erreur : Échec de la lecture des résultats." << endl;
        return EXIT_FAILURE;
    }

    // 7. Save and Cleanup
    char nom_fichier[256];
    generer_nom_fichier_resultat(nom_fichier, sizeof(nom_fichier), "res/opencl", "add", is_float, N);
    sauvegarder_matrice_csv(nom_fichier, h_result, N, is_float);

    // Cleanup OpenCL Resources
    clReleaseMemObject(d_mat1);
    clReleaseMemObject(d_mat2);
    clReleaseMemObject(d_result);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_result);  // Free allocated memory

    return EXIT_SUCCESS;
}