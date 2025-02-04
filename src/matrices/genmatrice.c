#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define MAX_N 2048  // Taille maximale de la matrice carrée
#define MIN_VAL -100
#define MAX_VAL 100

typedef float matrix_float_type;
typedef int matrix_int_type;

// Fonction pour mesurer le temps en millisecondes
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec) * 1000.0 + (tv.tv_usec) / 1000.0;
}

// ==================== Génération de matrices ====================
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

// ==================== Sauvegarde CSV ====================
void sauvegarder_matrice_csv(const char *filename, void *matrice, int taille, int is_float) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Erreur lors de l'ouverture du fichier %s\n", filename);
        return;
    }
    
    fprintf(file, "%d\n", taille);
    
    for (int i = 0; i < taille; i++) {
        for (int j = 0; j < taille; j++) {
            if (is_float) {
                fprintf(file, "%.6f", ((matrix_float_type*)matrice)[i * taille + j]);
            } else {
                fprintf(file, "%d", ((matrix_int_type*)matrice)[i * taille + j]);
            }
            
            if (j < taille - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
    // printf("Matrice de taille %dx%d sauvegardée dans %s\n", taille, taille, filename);
}

// ==================== Programme principal ====================
int main() {
    srand(time(NULL));

    // Boucle pour générer des matrices de taille 1x1 à MAX_N x MAX_N
    for (int taille = 1; taille <= MAX_N; taille *= 2) {
        
        // Allocation dynamique
        matrix_int_type *matrice_int = (matrix_int_type*)malloc(taille * taille * sizeof(matrix_int_type));
        matrix_float_type *matrice_float = (matrix_float_type*)malloc(taille * taille * sizeof(matrix_float_type));

        for (int j = 0; j < 2 ; j++) {
            // Génération des matrices aléatoires
            generer_matrice_cpu_int(matrice_int, taille);
            generer_matrice_cpu_float(matrice_float, taille);

            // Création des noms de fichiers dynamiques
            char filename_int[50], filename_float[50];
            sprintf(filename_int, "int/%dx%d-number%d.csv", taille, taille, j + 1);
            sprintf(filename_float, "float/%dx%d-number%d.csv", taille, taille, j + 1);

            // Sauvegarde des matrices
            sauvegarder_matrice_csv(filename_int, matrice_int, taille, 0);
            sauvegarder_matrice_csv(filename_float, matrice_float, taille, 1);
            
        }

        // Libération de la mémoire
        free(matrice_int);
        free(matrice_float);
    }

    return 0;
}   