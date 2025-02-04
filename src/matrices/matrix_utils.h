#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define MAX_N 2048  // Taille maximale de la matrice carrée
#define MIN_VAL -100
#define MAX_VAL 100

typedef float matrix_float_type;
typedef int matrix_int_type;

void generer_matrice_cpu_int(matrix_int_type *matrice, int taille);
void generer_matrice_cpu_float(matrix_float_type *matrice, int taille);
void sauvegarder_matrice_csv(const char *filename, void *matrice, int taille, int is_float);
void charger_matrice_csv(const char *filename, void **matrice, int *taille, int is_float);
