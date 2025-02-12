#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../matrices/matrix_utils.h"

typedef struct {
    int taille;
    int elements[MAX_N][MAX_N];
} Matrice;

void additionner_matrices(Matrice matrice1, Matrice matrice2, Matrice *resultat) {
    resultat->taille = matrice1.taille;
    for (int i = 0; i < matrice1.taille; i++) {
        for (int j = 0; j < matrice1.taille; j++) {
            resultat->elements[i][j] = matrice1.elements[i][j] + matrice2.elements[i][j];
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
    
    int is_float = type_matrice(argv[1]) || type_matrice(argv[2]);
    
    charger_matrice_csv(argv[1], &data1, &taille1, is_float);
    charger_matrice_csv(argv[2], &data2, &taille2, is_float);
    
    if (taille1 != taille2) {
        printf("Erreur : Les matrices doivent avoir la même taille pour être additionnées.\n");
        return EXIT_FAILURE;
    }
    
    matrice1.taille = matrice2.taille = taille1;
    for (int i = 0; i < taille1; i++) {
        for (int j = 0; j < taille1; j++) {
            matrice1.elements[i][j] = ((matrix_int_type*)data1)[i * taille1 + j];
            matrice2.elements[i][j] = ((matrix_int_type*)data2)[i * taille1 + j];
        }
    }
    
    free(data1);
    free(data2);
    
    // clock_t start = clock();
    additionner_matrices(matrice1, matrice2, &resultat);
    // clock_t end = clock();
    
    // double temps_execution = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    // printf("Addition terminée en %.2f ms.\n", temps_execution);

    return EXIT_SUCCESS;
}
