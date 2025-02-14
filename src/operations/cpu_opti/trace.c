#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../../matrices/matrix_utils.h"

#define BLOCK_SIZE 512  // Taille maximale d'un bloc

typedef enum { TYPE_INT, TYPE_FLOAT } TypeMatrice;

double calculer_trace_bloc(const void *bloc, int debut, int taille, int is_float) {
    double trace = 0.0;
    for (int i = 0; i < BLOCK_SIZE && (debut + i) < taille; i++) {
        if (is_float) {
            float *mat = (float *)bloc;
            trace += mat[(debut + i) * taille + debut + i];
        } else {
            int *mat = (int *)bloc;
            trace += mat[(debut + i) * taille + debut + i];
        }
    }
    return trace;
}

double calculer_trace_matrice(const void *data, int taille, int is_float) {
    double trace_totale = 0.0;
    for (int i = 0; i < taille; i += BLOCK_SIZE) {
        trace_totale += calculer_trace_bloc(data, i, taille, is_float);
    }
    return trace_totale;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Utilisation : %s <fichier_matrice.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int taille;
    void *data = NULL;
    
    int is_float = type_matrice(argv[1]);
    charger_matrice_csv(argv[1], &data, &taille, is_float);
    
    if (taille > MAX_N) {
        printf("Erreur : Taille de matrice trop grande (max %d)\n", MAX_N);
        free(data);
        return EXIT_FAILURE;
    }

    //clock_t start = clock();
    calculer_trace_matrice(data, taille, is_float);
    //clock_t end = clock();
    /*
    double temps_execution = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Trace de la matrice : %.2f\n", trace);
    printf("Calcul terminé en %.2f ms.\n", temps_execution);
    */
    free(data);
    return EXIT_SUCCESS;
}