#include <iostream>
#include <fstream>
#include <vector>
#include <CL/cl.hpp>
#include "../../matrices/matrix_utils.h"

using namespace std;

const char *KERNEL_FILE = "add.cl";

string readKernelFile(const char *filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Erreur : Impossible de lire le fichier du kernel OpenCL." << endl;
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

    int N = taille1;
    int matrixSize = N * N * (is_float ? sizeof(float) : sizeof(int));

    cl::Platform platform = cl::Platform::getDefault();
    cl::Device device = platform.getDevices(CL_DEVICE_TYPE_GPU)[0];
    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    string kernelSource = readKernelFile(KERNEL_FILE);
    cl::Program program(context, kernelSource);
    program.build({device});

    cl::Kernel kernel(program, is_float ? "add_matrices_float" : "add_matrices_int");
    cl::Buffer d_mat1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize, h_mat1);
    cl::Buffer d_mat2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, matrixSize, h_mat2);
    cl::Buffer d_result(context, CL_MEM_WRITE_ONLY, matrixSize);

    kernel.setArg(0, d_mat1);
    kernel.setArg(1, d_mat2);
    kernel.setArg(2, d_result);
    kernel.setArg(3, N);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N * N), cl::NDRange(256));
    queue.enqueueReadBuffer(d_result, CL_TRUE, 0, matrixSize, h_result);

    sauvegarder_matrice_csv("res/opencl_add.csv", h_result, N, is_float);
    return EXIT_SUCCESS;
}
