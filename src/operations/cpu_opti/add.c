#include "../../matrices/matrix_utils.h"

typedef enum { TYPE_INT, TYPE_FLOAT } TypeMatrice;
#define BLOCK_SIZE 512  // Taille maximale d'un bloc qui fonctionne sur votre système

typedef struct {
    int taille;
    TypeMatrice type;
    union {
        int elements_int[BLOCK_SIZE][BLOCK_SIZE];
        float elements_float[BLOCK_SIZE][BLOCK_SIZE];
    };
} MatriceBlock;

void additionner_bloc(const void *bloc1, const void *bloc2, void *resultat, 
                    int debut_i, int debut_j, int taille, int taille_totale, 
                    int is_float) {
    for(int i = 0; i < BLOCK_SIZE && (debut_i + i) < taille; i++) {
        for(int j = 0; j < BLOCK_SIZE && (debut_j + j) < taille; j++) {
            if(is_float) {
                float *b1 = (float*)bloc1;
                float *b2 = (float*)bloc2;
                float *res = (float*)resultat;
                res[(debut_i + i) * taille_totale + debut_j + j] = 
                    b1[(debut_i + i) * taille_totale + debut_j + j] + 
                    b2[(debut_i + i) * taille_totale + debut_j + j];
            } else {
                int *b1 = (int*)bloc1;
                int *b2 = (int*)bloc2;
                int *res = (int*)resultat;
                res[(debut_i + i) * taille_totale + debut_j + j] = 
                    b1[(debut_i + i) * taille_totale + debut_j + j] + 
                    b2[(debut_i + i) * taille_totale + debut_j + j];
            }
        }
    }
}

void additionner_matrices_grandes(const void *data1, const void *data2, void *resultat,
                                int taille, int is_float) {
    for(int i = 0; i < taille; i += BLOCK_SIZE) {
        for(int j = 0; j < taille; j += BLOCK_SIZE) {
            additionner_bloc(data1, data2, resultat, i, j, taille, taille, is_float);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Utilisation : %s <fichier_matrice1.csv> <fichier_matrice2.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int taille1, taille2;
    void *data1 = NULL, *data2 = NULL;
    void *resultat = NULL;
    
    int is_float1 = type_matrice(argv[1]);
    int is_float2 = type_matrice(argv[2]);
    
    if (is_float1 != is_float2) {
        printf("Erreur : Les matrices doivent être du même type.\n");
        return EXIT_FAILURE;
    }

    charger_matrice_csv(argv[1], &data1, &taille1, is_float1);
    charger_matrice_csv(argv[2], &data2, &taille2, is_float2);

    if (taille1 != taille2) {
        printf("Erreur : Les matrices doivent avoir la même taille.\n");
        free(data1);
        free(data2);
        return EXIT_FAILURE;
    }

    if (taille1 > MAX_N) {
        printf("Erreur : Taille de matrice trop grande (max %d)\n", MAX_N);
        free(data1);
        free(data2);
        return EXIT_FAILURE;
    }

    // Allouer l'espace pour le résultat
    size_t taille_mem = taille1 * taille1 * (is_float1 ? sizeof(float) : sizeof(int));
    resultat = malloc(taille_mem);
    if (!resultat) {
        printf("Erreur d'allocation mémoire\n");
        free(data1);
        free(data2);
        return EXIT_FAILURE;
    }

    clock_t start = clock();
    additionner_matrices_grandes(data1, data2, resultat, taille1, is_float1);
    clock_t end = clock();
    
    double temps_execution = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Addition terminée en %.2f ms.\n", temps_execution);

    // Sauvegarder le résultat si nécessaire
    // char nom_fichier_resultat[256];
    // generer_nom_fichier_resultat(nom_fichier_resultat, sizeof(nom_fichier_resultat),
    //                             "cpu", "addition", is_float1, taille1);
    // sauvegarder_matrice_csv(nom_fichier_resultat, resultat, taille1, is_float1);

    free(data1);
    free(data2);
    free(resultat);

    return EXIT_SUCCESS;
}
