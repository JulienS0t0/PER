#include "matrix_utils.h"

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

// ==================== Charger CSV ====================
void charger_matrice_csv(const char *filename, void **matrice, int *taille, int is_float) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Erreur lors de l'ouverture du fichier %s\n", filename);
        return;
    }

    // Lire la taille de la matrice
    if (fscanf(file, "%d", taille) != 1) {
        printf("Erreur de lecture de la taille de la matrice %s\n", filename);
        fclose(file);
        return;
    }

    // Allouer dynamiquement la matrice
    if (is_float) {
        *matrice = malloc((*taille) * (*taille) * sizeof(matrix_float_type));
    } else {
        *matrice = malloc((*taille) * (*taille) * sizeof(matrix_int_type));
    }

    // Lire les éléments de la matrice
    for (int i = 0; i < *taille; i++) {
        for (int j = 0; j < *taille; j++) {
            if (is_float) {
                if (fscanf(file, "%f,", &((matrix_float_type*)*matrice)[i * (*taille) + j]) != 1) {
                    printf("Erreur de lecture d'un élément float dans %s\n", filename);
                    fclose(file);
                    return;
                }
            } else {
                if (fscanf(file, "%d,", &((matrix_int_type*)*matrice)[i * (*taille) + j]) != 1) {
                    printf("Erreur de lecture d'un élément int dans %s\n", filename);
                    fclose(file);
                    return;
                }
            }
        }
    }

    fclose(file);
    printf("Matrice chargée depuis %s\n", filename);
}