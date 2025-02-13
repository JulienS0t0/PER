#include "../../matrices/matrix_utils.h"
#include <stdio.h>
#include <stdlib.h>

typedef enum { TYPE_INT, TYPE_FLOAT } TypeMatrice;
#define BLOCK_SIZE 512  // Taille d'un bloc de traitement

typedef struct {
    int taille;
    TypeMatrice type;
    union {
        int elements_int[BLOCK_SIZE][BLOCK_SIZE];
        float elements_float[BLOCK_SIZE][BLOCK_SIZE];
    };
} MatriceBlock;

// Fonction pour calculer la trace en utilisant des blocs (tiling)
double trace_matrice_blocs(MatriceBlock *mat, int taille, int is_float) {
    double trace = 0.0;

    if (is_float) {
        float (*m)[BLOCK_SIZE] = mat->elements_float;

        // Parcours par blocs pour améliorer la gestion du cache
        for (int i = 0; i < taille; i += BLOCK_SIZE) {
            for (int j = i; j < i + BLOCK_SIZE && j < taille; j++) {
                trace += m[j][j];
            }
        }
    } else {
        int (*m)[BLOCK_SIZE] = mat->elements_int;

        for (int i = 0; i < taille; i += BLOCK_SIZE) {
            for (int j = i; j < i + BLOCK_SIZE && j < taille; j++) {
                trace += m[j][j];
            }
        }
    }

    return trace;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Utilisation : %s <fichier_matrice.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int taille;
    MatriceBlock matrice;

    int is_float = type_matrice(argv[1]);

    charger_matrice_csv(argv[1], &matrice, &taille, is_float);

    if (taille > MAX_N) {
        printf("Erreur : Taille de matrice trop grande (max %d)\n", MAX_N);
        return EXIT_FAILURE;
    }

    clock_t start = clock();
    double trace = trace_matrice_blocs(&matrice, taille, is_float);
    clock_t end = clock();

    double temps_execution = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Trace calculée en %.2f ms.\n", temps_execution);
    printf("Trace : %.6f\n", trace);

    return EXIT_SUCCESS;
}
