#include "matrix_utils.h"

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
            char filename_int[75], filename_float[75];
            sprintf(filename_int, "../../out/matrices/int/%dx%d-number%d.csv", taille, taille, j + 1);
            sprintf(filename_float, "../../out/matrices/float/%dx%d-number%d.csv", taille, taille, j + 1);

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