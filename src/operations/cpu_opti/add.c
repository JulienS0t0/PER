#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../matrices/matrix_utils.h"

typedef enum { TYPE_INT, TYPE_FLOAT } TypeMatrice;

typedef struct {
    int taille;
    TypeMatrice type;
    union {
        int elements_int[MAX_N][MAX_N];
        float elements_float[MAX_N][MAX_N];
    };
} Matrice;

void additionner_matrices(const Matrice *matrice1, const Matrice *matrice2, Matrice *resultat) {
    int taille = matrice1->taille;
    
    if (matrice1->type != matrice2->type) {
        printf("Erreur : Les matrices doivent être du même type pour être additionnées.\n");
        exit(EXIT_FAILURE);
    }

    resultat->taille = taille;
    resultat->type = matrice1->type;

    for (int i = 0; i < taille; i++) {
        for (int j = 0; j < taille; j++) {
            if (matrice1->type == TYPE_INT) {
                resultat->elements_int[i][j] = matrice1->elements_int[i][j] + matrice2->elements_int[i][j];
            } else {
                resultat->elements_float[i][j] = matrice1->elements_float[i][j] + matrice2->elements_float[i][j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Utilisation : %s <fichier_matrice1.csv> <fichier_matrice2.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    Matrice matrice1, matrice2, resultat;
    int taille1, taille2;
    void *data1 = NULL, *data2 = NULL;
    
    int is_float1 = type_matrice(argv[1]);
    int is_float2 = type_matrice(argv[2]);

    if (is_float1 != is_float2) {
        printf("Erreur : Les matrices doivent être du même type pour être additionnées.\n");
        return EXIT_FAILURE;
    }

    charger_matrice_csv(argv[1], &data1, &taille1, is_float1);
    charger_matrice_csv(argv[2], &data2, &taille2, is_float2);
    
    if (taille1 != taille2) {
        printf("Erreur : Les matrices doivent avoir la même taille pour être additionnées.\n");
        free(data1);
        free(data2);
        return EXIT_FAILURE;
    }
    
    matrice1.taille = matrice2.taille = taille1;
    matrice1.type = matrice2.type = is_float1 ? TYPE_FLOAT : TYPE_INT;

    if (is_float1) {
        float *data_float1 = (float *)data1;
        float *data_float2 = (float *)data2;
        for (int i = 0; i < taille1; i++) {
            for (int j = 0; j < taille1; j++) {
                matrice1.elements_float[i][j] = data_float1[i * taille1 + j];
                matrice2.elements_float[i][j] = data_float2[i * taille1 + j];
            }
        }
    } else {
        int *data_int1 = (int *)data1;
        int *data_int2 = (int *)data2;
        for (int i = 0; i < taille1; i++) {
            for (int j = 0; j < taille1; j++) {
                matrice1.elements_int[i][j] = data_int1[i * taille1 + j];
                matrice2.elements_int[i][j] = data_int2[i * taille1 + j];
            }
        }
    }

    free(data1);
    free(data2);
    
    clock_t start = clock();
    additionner_matrices(&matrice1, &matrice2, &resultat);
    clock_t end = clock();

    double temps_execution = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Addition terminée en %.2f ms.\n", temps_execution);

    return EXIT_SUCCESS;
}
