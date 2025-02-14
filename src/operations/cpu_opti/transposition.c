#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../matrices/matrix_utils.h"

#define BLOCK_SIZE 512

typedef enum { TYPE_INT, TYPE_FLOAT } TypeMatrice;

typedef struct {
    int taille;
    TypeMatrice type;
    union {
        int elements_int[BLOCK_SIZE][BLOCK_SIZE];
        float elements_float[BLOCK_SIZE][BLOCK_SIZE];
    };
} MatriceBlock;

void transposer_bloc(const void *src, void *dest, int debut_i, int debut_j, int taille, int taille_totale, int is_float) {
    for (int i = 0; i < BLOCK_SIZE && (debut_i + i) < taille; i++) {
        for (int j = 0; j < BLOCK_SIZE && (debut_j + j) < taille; j++) {
            if (is_float) {
                float *s = (float*)src;
                float *d = (float*)dest;
                d[(debut_j + j) * taille_totale + (debut_i + i)] = 
                    s[(debut_i + i) * taille_totale + (debut_j + j)];
            } else {
                int *s = (int*)src;
                int *d = (int*)dest;
                d[(debut_j + j) * taille_totale + (debut_i + i)] = 
                    s[(debut_i + i) * taille_totale + (debut_j + j)];
            }
        }
    }
}

void transposer_matrice(const void *src, void *dest, int taille, int is_float) {
    for (int i = 0; i < taille; i += BLOCK_SIZE) {
        for (int j = 0; j < taille; j += BLOCK_SIZE) {
            transposer_bloc(src, dest, i, j, taille, taille, is_float);
        }
    }
}

int main(int argc, char *argv[]) {
     if (argc < 3) {
        printf("Utilisation : %s <fichier_matrice1.csv> [unused] [chemin_stockage]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int taille;
    void *data = NULL;
    void *resultat = NULL;
    const char *chemin_stockage = (argc > 3) ? argv[3] : NULL;
    
    int is_float = type_matrice(argv[1]);
    charger_matrice_csv(argv[1], &data, &taille, is_float);

    if (taille > MAX_N) {
        printf("Erreur : Taille de matrice trop grande (max %d)\n", MAX_N);
        free(data);
        return EXIT_FAILURE;
    }

    size_t taille_mem = taille * taille * (is_float ? sizeof(float) : sizeof(int));
    resultat = malloc(taille_mem);
    if (!resultat) {
        printf("Erreur d'allocation mémoire\n");
        free(data);
        return EXIT_FAILURE;
    }

    clock_t start = clock();
    transposer_matrice(data, resultat, taille, is_float);
    clock_t end = clock();
    
    double temps_execution = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Transposition terminée en %.2f ms.\n", temps_execution);

    if (chemin_stockage) {
        sauvegarder_matrice_csv(chemin_stockage, resultat, taille, is_float);
        printf("Résultat enregistré dans le fichier : %s", chemin_stockage);
    }

    free(data);
    free(resultat);
    
    return EXIT_SUCCESS;
}
