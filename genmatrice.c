    #include <stdio.h>
    #include <stdlib.h>
    #include <time.h>
    #include <cuda_runtime.h>
    #include <CL/cl.h>
    #include <sys/time.h>

    #define MAX_N 1024  // Taille maximale de la matrice carrée
    #define MIN_VAL 0
    #define MAX_VAL 100

    typedef float matrix_float_type;
    typedef int matrix_int_type;

    // Fonction pour mesurer le temps en millisecondes
    double get_time() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return (tv.tv_sec) * 1000.0 + (tv.tv_usec) / 1000.0;
    }

    // Fonction pour écrire les résultats dans un fichier CSV
    void enregistrer_resultats_csv(const char *filename, int taille, double cpu_time, double cuda_time, double opencl_time) {
        FILE *file = fopen(filename, "a");
        if (file == NULL) {
            printf("Erreur lors de l'ouverture du fichier CSV.\n");
            return;
        }
        fprintf(file, "%d,%.6f,%.6f,%.6f\n", taille, cpu_time, cuda_time, opencl_time);
        fclose(file);
    }

    // ==================== CPU Version ====================
    void generer_matrice_cpu_int(matrix_int_type *matrice, int taille) {
        for (int i = 0; i < taille * taille; i++) {
            matrice[i] = rand() % (MAX_VAL - MIN_VAL + 1) + MIN_VAL;
        }
    }

    void generer_matrice_cpu_float(matrix_float_type *matrice, int taille) {
        for (int i = 0; i < taille * taille; i++) {
            matrice[i] = ((float)rand() / RAND_MAX) * (MAX_VAL - MIN_VAL) + MIN_VAL;
        }
    }

    // ==================== Main Function ====================
    int main() {
        srand(time(NULL));
        FILE *file_int = fopen("benchmark_genmatrice_results_int.csv", "w");
        FILE *file_float = fopen("benchmark_genmatrice_results_float.csv", "w");
        fprintf(file_int, "Taille,CPU (ms),CUDA (ms),OpenCL (ms)\n");
        fprintf(file_float, "Taille,CPU (ms),CUDA (ms),OpenCL (ms)\n");
        fclose(file_int);
        fclose(file_float);
        
        for (int taille = 1; taille <= MAX_N; taille *= 2) {
            matrix_int_type *matrice_int = (matrix_int_type*)malloc(taille * taille * sizeof(matrix_int_type));
            matrix_float_type *matrice_float = (matrix_float_type*)malloc(taille * taille * sizeof(matrix_float_type));

            double start, end, cpu_time_int, cpu_time_float;

            // Benchmark CPU pour int
            start = get_time();
            generer_matrice_cpu_int(matrice_int, taille);
            end = get_time();
            cpu_time_int = end - start;

            // Benchmark CPU pour float
            start = get_time();
            generer_matrice_cpu_float(matrice_float, taille);
            end = get_time();
            cpu_time_float = end - start;

            enregistrer_resultats_csv("benchmark_genmatrice_results_int.csv", taille, cpu_time_int, 0.0, 0.0);
            enregistrer_resultats_csv("benchmark_genmatrice_results_float.csv", taille, cpu_time_float, 0.0, 0.0);

            free(matrice_int);
            free(matrice_float);
        }
        return 0;
    }
