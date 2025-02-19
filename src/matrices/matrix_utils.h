#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <sys/stat.h>  // Pour mkdir
#include <sys/types.h>
#include <sys/time.h>

#define MAX_N 8192// 16384 // 32768 // Taille maximale de la matrice carrée
#define MIN_VAL -100
#define MAX_VAL 100

typedef float matrix_float_type;
typedef int matrix_int_type;

void generer_matrice_cpu_int(matrix_int_type *matrice, int taille);
void generer_matrice_cpu_float(matrix_float_type *matrice, int taille);
void sauvegarder_matrice_csv(const char *filename, void *matrice, int taille, int is_float);
void charger_matrice_csv(const char *filename, void **matrice, int *taille, int is_float);
void obtenir_date_heure(char *buffer, size_t buffer_size);
void generer_nom_fichier_resultat(char *buffer, size_t buffer_size, const char *operationtype, const char *operation, int is_float, int taille);
int type_matrice(const char *filename);
