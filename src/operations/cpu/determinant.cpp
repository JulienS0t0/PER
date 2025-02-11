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
void determinantMatrice(T *matrice1, T *resultat, int taille) {
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
            *resultat = 0;
            return;
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

    *resultat = det; // Stocke le résultat final dans la variable passée en argument
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Utilisation : " << argv[0] << " <fichier_matrice1.csv>" << endl;
        return EXIT_FAILURE;
    }

    const char *fichier1 = argv[1];

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
        determinantMatrice((matrix_float_type*)matrice1, (matrix_float_type*)resultat, taille1);
    } else {
        determinantMatrice((matrix_int_type*)matrice1, (matrix_int_type*)resultat, taille1);
    }

    // auto stop = high_resolution_clock::now();
    // auto duration = duration_cast<milliseconds>(stop - start);

    // cout << "Trace terminée en " << duration.count() << " ms." << endl;

    // // Génération du nom de fichier
    // char nom_fichier[256];
    // generer_nom_fichier_resultat(nom_fichier, sizeof(nom_fichier), "res/cpu", "determinant", is_float, taille);
    // // Sauvegarder la matrice résultante
    // sauvegarder_matrice_csv(nom_fichier, resultat, taille, is_float);
    // cout << "Résultat enregistré dans le fichier : " << nom_fichier << endl;

    // Libération de la mémoire
    free(matrice1);
    free(resultat);
    return EXIT_SUCCESS;
}
