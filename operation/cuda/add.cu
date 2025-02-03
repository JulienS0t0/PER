  #include <fstream>
  #include <sstream>
  #include <vector>
  #include <iostream>
  #include <cmath>
  #include <chrono>
  #include <cuda_runtime.h>

  // Function to read a matrix from a CSV file
  void readCSV(const std::string &filename, float *matrix, int N) {
    std::ifstream file(filename);
    std::string line;
    int i = 0;
    while (std::getline(file, line) && i < N * N) {
      std::stringstream ss(line);
      std::string value;
      while (std::getline(ss, value, ',') && i < N * N) {
        matrix[i++] = std::stof(value);
      }
    }
  }

  // CUDA kernel to add two matrices
  __global__
  void add(int N, float *x, float *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N * N) {
      y[index] = x[index] + y[index];
    }
  }

  int main(int argc, char *argv[]) {
    if (argc != 3) {
      std::cerr << "Usage: " << argv[0] << " <path_to_first_matrix> <path_to_second_matrix>" << std::endl;
      return 1;
    }

    std::string path1 = argv[1];
    std::string path2 = argv[2];

    int N = 8; // Assuming N is the dimension of the matrix (N x N)
    float *x, *y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * N * sizeof(float));
    cudaMallocManaged(&y, N * N * sizeof(float));

    // Read matrices from CSV files
    readCSV(path1, x, N);
    readCSV(path2, y, N);

    // Measure the time of the addition
    auto start = std::chrono::high_resolution_clock::now();

    // Run kernel on N x N elements on the GPU
    int blockSize = 256;
    int numBlocks = (N * N + blockSize - 1) / blockSize;
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Log the duration to a file
    std::ofstream logFile("../../res/test.log", std::ios_base::app);
    logFile << "Addition duration: " << duration.count() << " seconds" << std::endl;
    logFile.close();

    // Check for errors (all values should be the sum of corresponding elements)
    float maxError = 0.0f;
    for (int i = 0; i < N * N; i++)
      maxError = fmax(maxError, fabs(y[i] - (x[i] + y[i])));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    
    return 0;
  }
