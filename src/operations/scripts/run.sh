#!/usr/bin/env bash

# Usage : run.sh operation
# This file should be called from the root of the project
# Please use it via the make file 'make run operation'

if [ -z "$1" ]; then
    echo "Usage: $0 <operation>"
    exit 1
fi

OPERATION=$1
MATRIX_DIR="out/matrices"
OPERATIONS_DIR="out/operations"
RESULTS_DIR="res"
CUDA_RESULTS_DIR="$RESULTS_DIR/cuda"
OPENCL_RESULTS_DIR="$RESULTS_DIR/opencl"

mkdir -p "$CUDA_RESULTS_DIR"
mkdir -p "$OPENCL_RESULTS_DIR"

# Obtenir la date et l'heure actuelles
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Trouver tous les types de matrices (int, float, etc.)
for TYPE in $(ls "$MATRIX_DIR"); do
    echo "Processing matrices of type: $TYPE"
    
    # Récupérer toutes les matrices paires (number1 et number2)
    for FILE1 in "$MATRIX_DIR/$TYPE"/*-number1.csv; do
        FILE2="${FILE1/-number1.csv/-number2.csv}"
        
        if [ ! -f "$FILE2" ]; then
            echo "Skipping unmatched file: $FILE1"
            continue
        fi

        # Extraire la taille de la matrice depuis le nom du fichier
        MATRIX_SIZE=$(basename "$FILE1" | cut -d'-' -f1)

        # Exécuter chaque implémentation disponible de l'opération
        for IMPL in cpu cuda opencl; do
            EXECUTABLE="$OPERATIONS_DIR/$IMPL/$OPERATION"
            
            if [ -x "$EXECUTABLE" ]; then
                echo "Running $EXECUTABLE with $FILE1 and $FILE2"
                if [ "$IMPL" == "cuda" ]; then
                    nvprof --log-file "$CUDA_RESULTS_DIR/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}.log" "$EXECUTABLE" "$FILE1" "$FILE2"
                elif [ "$IMPL" == "opencl" ]; then
                    nsys profile -o "$OPENCL_RESULTS_DIR/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}" "$EXECUTABLE" "$FILE1" "$FILE2"
                else
                    # "$EXECUTABLE" "$FILE1" "$FILE2" # Run without monitoring
                    /usr/bin/time -v "$EXECUTABLE" "$FILE1" "$FILE2" 2> "res/cpu/${TIMESTAMP}${OPERATION}${TYPE}_${MATRIX_SIZE}.log"
                fi
            else
                echo "Executable not found: $EXECUTABLE"
            fi
        done
    done

done