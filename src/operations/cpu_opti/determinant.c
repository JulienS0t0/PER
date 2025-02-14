#include "../../matrices/matrix_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef enum { TYPE_INT, TYPE_FLOAT } TypeMatrice;
#define BLOCK_SIZE 512  // Taille d'un bloc pour travailler efficacement en mémoire

typedef struct {
    int taille;
    TypeMatrice type;
    union {
        int elements_int[BLOCK_SIZE][BLOCK_SIZE];
        float elements_float[BLOCK_SIZE][BLOCK_SIZE];
    };
} MatriceBlock;

double determinant_blocs(void *matrice, int taille, int is_float) {
    int swap_count = 0;  // Compteur de permutations de lignes (affecte le signe du déterminant)
    double det = 1.0;

    if (is_float) {
        float (*m)[taille] = (float (*)[taille])matrice;

        for (int i = 0; i < taille; i += BLOCK_SIZE) {
            for (int j = i; j < taille; j++) {
                // Sélection du pivot
                int pivot_index = j;
                for (int k = j + 1; k < taille; k++) {
                    if (fabs(m[k][j]) > fabs(m[pivot_index][j])) {
                        pivot_index = k;
                    }
                }

                // Échange de lignes si nécessaire
                if (pivot_index != j) {
                    for (int k = 0; k < taille; k++) {
                        float temp = m[j][k];
                        m[j][k] = m[pivot_index][k];
                        m[pivot_index][k] = temp;
                    }
                    swap_count++;
                }

                // Vérifier si la matrice est singulière
                if (m[j][j] == 0.0) {
                    return 0.0;
                }

                // Élimination de Gauss en bloc
                for (int k = j + 1; k < taille && k < j + BLOCK_SIZE; k++) {
                    float coef = m[k][j] / m[j][j];
                    for (int l = j; l < taille; l++) {
                        m[k][l] -= coef * m[j][l];
                    }
                }
            }
        }

        // Produit des éléments diagonaux
        for (int i = 0; i < taille; i++) {
            det *= m[i][i];
        }
    } else {
        int (*m)[taille] = (int (*)[taille])matrice;

        for (int i = 0; i < taille; i += BLOCK_SIZE) {
            for (int j = i; j < taille; j++) {
                // Sélection du pivot
                int pivot_index = j;
                for (int k = j + 1; k < taille; k++) {
                    if (abs(m[k][j]) > abs(m[pivot_index][j])) {
                        pivot_index = k;
                    }
                }

                // Échange de lignes si nécessaire
                if (pivot_index != j) {
                    for (int k = 0; k < taille; k++) {
                        int temp = m[j][k];
                        m[j][k] = m[pivot_index][k];
                        m[pivot_index][k] = temp;
                    }
                    swap_count++;
                }

                // Vérifier si la matrice est singulière
                if (m[j][j] == 0) {
                    return 0;
                }

                // Élimination de Gauss en bloc
                for (int k = j + 1; k < taille && k < j + BLOCK_SIZE; k++) {
                    int coef = m[k][j] / m[j][j];
                    for (int l = j; l < taille; l++) {
                        m[k][l] -= coef * m[j][l];
                    }
                }
            }
        }

        // Produit des éléments diagonaux
        det = 1;
        for (int i = 0; i < taille; i++) {
            det *= m[i][i];
        }
    }

    return (swap_count % 2 == 0) ? det : -det;  // Inversion du signe selon les permutations
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Utilisation : %s <fichier_matrice.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int taille;
    static int data_int[BLOCK_SIZE][BLOCK_SIZE];  // Stocke les matrices int en statique
    static float data_float[BLOCK_SIZE][BLOCK_SIZE];  // Stocke les matrices float en statique
    void *data = NULL;

    int is_float = type_matrice(argv[1]);

    // Charger la matrice directement dans un tableau statique
    if (is_float) {
        data = data_float;
    } else {
        data = data_int;
    }

    charger_matrice_csv(argv[1], &data, &taille, is_float);

    if (taille > MAX_N) {
        printf("Erreur : Taille de matrice trop grande (max %d)\n", MAX_N);
        return EXIT_FAILURE;
    }

    //clock_t start = clock();
    determinant_blocs(data, taille, is_float);
    //clock_t end = clock();
    /*

    double temps_execution = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Déterminant calculé en %.2f ms.\n", temps_execution);
    printf("Déterminant : %.6f\n", det);*/

    return EXIT_SUCCESS;
}
