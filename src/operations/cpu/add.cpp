#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "../../matrices/matrix_utils.h"

using namespace std;
using namespace std::chrono;

template <typename T>
void additionnerMatrices(T *matrice1, T *matrice2, T *resultat, int taille) {
    for (int i = 0; i < taille * taille; i++) {
        resultat[i] = matrice1[i] + matrice2[i];
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Utilisation : " << argv[0] << " <fichier_matrice1.csv> <fichier_matrice2.csv> [chemin_stockage]" << endl;
        return EXIT_FAILURE;
    }

    const char *fichier1 = argv[1];
    const char *fichier2 = argv[2];
    const char *chemin_stockage = (argc > 3) ? argv[3] : nullptr;

    bool is_float = type_matrice(fichier1) || type_matrice(fichier2);

    int taille1, taille2;
    void *matrice1 = nullptr, *matrice2 = nullptr, *resultat = nullptr;

    // Charger les matrices
    charger_matrice_csv(fichier1, &matrice1, &taille1, is_float);
    charger_matrice_csv(fichier2, &matrice2, &taille2, is_float);

    // Vérifier que les tailles correspondent
    if (taille1 != taille2) {
        cerr << "Erreur : Les matrices doivent avoir la même taille pour être additionnées." << endl;
        free(matrice1);
        free(matrice2);
        return EXIT_FAILURE;
    }

    int taille = taille1;

    if (is_float) {
        resultat = malloc(taille * taille * sizeof(matrix_float_type));
    } else {
        resultat = malloc(taille * taille * sizeof(matrix_int_type));
    }

    if (!resultat) {
        cerr << "Erreur d'allocation mémoire pour la matrice résultat" << endl;
        free(matrice1);
        free(matrice2);
        return EXIT_FAILURE;
    }

    // Mesurer le temps d'exécution
    auto start = high_resolution_clock::now();
    
    if (is_float) {
        additionnerMatrices((matrix_float_type*)matrice1, (matrix_float_type*)matrice2, (matrix_float_type*)resultat, taille);
    } else {
        additionnerMatrices((matrix_int_type*)matrice1, (matrix_int_type*)matrice2, (matrix_int_type*)resultat, taille);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "Addition terminée en " << duration.count() << " ms." << endl;

    // Sauvegarde si un chemin de stockage est fourni
    if (chemin_stockage) {
        sauvegarder_matrice_csv(chemin_stockage, resultat, taille, is_float);
        cout << "Résultat enregistré dans le fichier : " << chemin_stockage << endl;
    }

    // Libération de la mémoire
    free(matrice1);
    free(matrice2);
    free(resultat);

    return EXIT_SUCCESS;
}
