#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "../../matrices/matrix_utils.h"

using namespace std;
using namespace std::chrono;

template <typename T>
void transpositionMatrice(T *matrice1, T *resultat, int taille) {
    for (int i = 0; i < taille; i++) {
        for (int j = 0; j < taille; j++) {
            resultat[j * taille + i] = matrice1[i * taille + j];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Utilisation : " << argv[0] << " <fichier_matrice1.csv> [unused] [chemin_stockage]" << endl;
        return EXIT_FAILURE;
    }

    const char *fichier1 = argv[1];
    const char *chemin_stockage = (argc > 3) ? argv[3] : nullptr;

    bool is_float = type_matrice(fichier1);

    int taille1;
    void *matrice1 = nullptr, *resultat = nullptr;

    // Charger les matrices
    charger_matrice_csv(fichier1, &matrice1, &taille1, is_float);

    if (is_float) {
        resultat = malloc(taille1 * taille1 * sizeof(matrix_float_type));
    } else {
        resultat = malloc(taille1 * taille1 * sizeof(matrix_int_type));
    }

    if (!resultat) {
        cerr << "Erreur d'allocation mémoire pour la matrice résultat" << endl;
        free(matrice1);
        return EXIT_FAILURE;
    }

    // Mesurer le temps d'exécution
    auto start = high_resolution_clock::now();
    
    if (is_float) {
        transpositionMatrice((matrix_float_type*)matrice1, (matrix_float_type*)resultat, taille1);
    } else {
        transpositionMatrice((matrix_int_type*)matrice1, (matrix_int_type*)resultat, taille1);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Transposition terminée en " << duration.count() << " ms." << endl;
    
    if (chemin_stockage){
        sauvegarder_matrice_csv(chemin_stockage, resultat, taille1, is_float);
    }

    free(matrice1);
    free(resultat);
    return EXIT_SUCCESS;
}
