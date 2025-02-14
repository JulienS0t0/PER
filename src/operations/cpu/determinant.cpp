#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>
#include "../../matrices/matrix_utils.h"

using namespace std;
using namespace std::chrono;

template <typename T>
T determinantMatrice(T *matrice1, int taille) {
    vector<vector<T>> mat(taille, vector<T>(taille));

    // Copier la matrice 1D en une structure 2D pour plus de lisibilité
    for (int i = 0; i < taille; i++) {
        for (int j = 0; j < taille; j++) {
            mat[i][j] = matrice1[i * taille + j];
        }
    }

    T det = 1;  // Initialisation du déterminant
    for (int i = 0; i < taille; i++) {
        // Vérification si le pivot est nul
        if (fabs(mat[i][i]) < 1e-9) {
            return 0;
        }

        // Réduction en forme triangulaire
        for (int j = i + 1; j < taille; j++) {
            T facteur = mat[j][i] / mat[i][i];
            for (int k = i; k < taille; k++) {
                mat[j][k] -= facteur * mat[i][k];
            }
        }

        // Multiplication des éléments diagonaux
        det *= mat[i][i];
    }

    return det; // Retourne le déterminant
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Utilisation : " << argv[0] << " <fichier_matrice1.csv>" << endl;
        return EXIT_FAILURE;
    }

    const char *fichier1 = argv[1];

    bool is_float = type_matrice(fichier1);

    int taille1;
    void *matrice1 = nullptr;

    // Charger la matrice
    charger_matrice_csv(fichier1, &matrice1, &taille1, is_float);

    if (!matrice1) {
        cerr << "Erreur de chargement de la matrice" << endl;
        return EXIT_FAILURE;
    }

    // Mesurer le temps d'exécution
    auto start = high_resolution_clock::now();

    double res;
    if (is_float) {
        res = determinantMatrice((matrix_float_type*)matrice1, taille1);
    } else {
        res = determinantMatrice((matrix_int_type*)matrice1, taille1);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Déterminant : " << res << endl;
    cout << "Calcul terminé en " << duration.count() << " ms." << endl;

    // Libération de la mémoire
    free(matrice1);
    return EXIT_SUCCESS;
}
