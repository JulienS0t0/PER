#include <iostream>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std;
using namespace std::chrono;

template <typename T>
void additionnerMatrices(T *matrice1, T *matrice2, T *resultat, int taille) {
  for (int i = 0; i < taille * taille; i++) {
    resultat[i] = matrice1[i] + matrice2[i];
  }
}

int main(void)
{
  const int taille = 1000; // Exemple de taille de matrice
  int *matrice1 = new int[taille * taille];
  int *matrice2 = new int[taille * taille];
  int *resultat = new int[taille * taille];

  // Initialisation des matrices avec des valeurs aléatoires
  for (int i = 0; i < taille * taille; i++) {
    matrice1[i] = rand() % 100;
    matrice2[i] = rand() % 100;
  }

  auto start = high_resolution_clock::now();
  additionnerMatrices(matrice1, matrice2, resultat, taille);
  auto end = high_resolution_clock::now();

  auto duration = duration_cast<milliseconds>(end - start).count();
  cout << "Temps d'addition des matrices: " << duration << " ms" << endl;

  delete[] matrice1;
  delete[] matrice2;
  delete[] resultat;

  return 0;
}
