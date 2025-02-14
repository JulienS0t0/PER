#include "../../matrices/matrix_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef enum { TYPE_INT, TYPE_FLOAT } TypeMatrice;
#define BLOCK_SIZE 512  // Taille d'un bloc pour une meilleure gestion mémoire

// Structure pour gérer une matrice
typedef struct {
    int taille;
    TypeMatrice type;
    void *elements;
} MatriceBlock;

double determinant_blocs(void *matrice, int taille, int is_float) {
    int swap_count = 0;  // Compteur de permutations de lignes
    double det = 1.0;

    if (is_float) {
        float (*m)[taille] = (float (*)[taille]) matrice;

        for (int j = 0; j < taille; j++) {
            int pivot_index = j;
            for (int k = j + 1; k < taille; k++) {
                if (fabs(m[k][j]) > fabs(m[pivot_index][j])) {
                    pivot_index = k;
                }
            }

            if (m[pivot_index][j] == 0.0) {
                return 0.0;
            }

            if (pivot_index != j) {
                for (int k = 0; k < taille; k++) {
                    float temp = m[j][k];
                    m[j][k] = m[pivot_index][k];
                    m[pivot_index][k] = temp;
                }
                swap_count++;
            }

            for (int k = j + 1; k < taille; k++) {
                float coef = m[k][j] / m[j][j];
                for (int l = j; l < taille; l++) {
                    m[k][l] -= coef * m[j][l];
                }
            }
        }

        for (int i = 0; i < taille; i++) {
            det *= m[i][i];
        }
    } else {
        int (*m)[taille] = (int (*)[taille]) matrice;

        for (int j = 0; j < taille; j++) {
            int pivot_index = j;
            for (int k = j + 1; k < taille; k++) {
                if (abs(m[k][j]) > abs(m[pivot_index][j])) {
                    pivot_index = k;
                }
            }

            if (m[pivot_index][j] == 0) {
                return 0;
            }

            if (pivot_index != j) {
                for (int k = 0; k < taille; k++) {
                    int temp = m[j][k];
                    m[j][k] = m[pivot_index][k];
                    m[pivot_index][k] = temp;
                }
                swap_count++;
            }

            for (int k = j + 1; k < taille; k++) {
                if (m[j][j] == 0) continue;  // Éviter la division par zéro
                int coef = m[k][j] / m[j][j];
                for (int l = j; l < taille; l++) {
                    m[k][l] -= coef * m[j][l];
                }
            }
        }

        for (int i = 0; i < taille; i++) {
            det *= m[i][i];
        }
    }

    return (swap_count % 2 == 0) ? det : -det;
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
    double res = determinant_blocs(data, taille, is_float);
    printf("Déterminant : %.6f\n", res);

    return EXIT_SUCCESS;
}
