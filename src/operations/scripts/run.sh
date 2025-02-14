#!/usr/bin/env bash

# Usage : run.sh operation
# This file should be called from the root of the project
# Please use it via the make file 'make run operation'

if [ -z "$1" ]; then
    echo "Usage: $0 <operation>"
    exit 1
fi

# Obtenir la date et l'heure actuelles
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

OPERATION=$1
MATRIX_DIR="out/matrices"
OPERATIONS_DIR="out/operations"
RESULTS_DIR="res/raw/${TIMESTAMP}_${OPERATION}"
CUDA_RESULTS_DIR="${RESULTS_DIR}/cuda"
OPENCL_RESULTS_DIR="${RESULTS_DIR}/opencl"
CPU_RESULTS_DIR="${RESULTS_DIR}/cpu"
CPU_OPTI_O2_RESULTS_DIR="${RESULTS_DIR}/cpu_opti_O2"
CPU_OPTI_O3_RESULTS_DIR="${RESULTS_DIR}/cpu_opti_O3"

# Détection du chemin de `time`
TIME_CMD=$(command -v time || echo "/usr/bin/time")

mkdir -p "${CUDA_RESULTS_DIR}"
mkdir -p "${OPENCL_RESULTS_DIR}"
mkdir -p "${CPU_RESULTS_DIR}"
mkdir -p "${CPU_OPTI_O2_RESULTS_DIR}"
mkdir -p "${CPU_OPTI_O3_RESULTS_DIR}"

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
        # for IMPL in cpu cuda opencl; do
        for IMPL in cpu cpu_opti_O2 cpu_opti_O3 cuda opencl; do
            EXECUTABLE="$OPERATIONS_DIR/$IMPL/$OPERATION"
            
            if [ -x "$EXECUTABLE" ]; then
                echo "Running $EXECUTABLE with $FILE1 and $FILE2"
                if [ "$IMPL" == "cuda" ]; then
                    nsys profile -o "$CUDA_RESULTS_DIR/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}" "$EXECUTABLE" "$FILE1" "$FILE2" &> "${CUDA_RESULTS_DIR}/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}.log"
                    #$TIME_CMD -v "$EXECUTABLE" "$FILE1" "$FILE2" &> "${CUDA_RESULTS_DIR}/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}.log"
                elif [ "$IMPL" == "opencl" ]; then
                    nsys profile -o "$OPENCL_RESULTS_DIR/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}" "$EXECUTABLE" "$FILE1" "$FILE2" &> "${OPENCL_RESULTS_DIR}/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}.log"
                    #$TIME_CMD -v "$EXECUTABLE" "$FILE1" "$FILE2" &> "${OPENCL_RESULTS_DIR}/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}.log"
                elif [ "$IMPL" == "cpu_opti_O2" ]; then
                    $TIME_CMD -v "$EXECUTABLE" "$FILE1" "$FILE2" &> "${CPU_OPTI_O2_RESULTS_DIR}/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}.log"
                elif [ "$IMPL" == "cpu_opti_O3" ]; then
                    $TIME_CMD -v "$EXECUTABLE" "$FILE1" "$FILE2" &> "${CPU_OPTI_O3_RESULTS_DIR}/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}.log"
                else
                    # "$EXECUTABLE" "$FILE1" "$FILE2" # Run without monitoring
                    $TIME_CMD -v "$EXECUTABLE" "$FILE1" "$FILE2" &> "${CPU_RESULTS_DIR}/${TIMESTAMP}_${OPERATION}_${TYPE}_${MATRIX_SIZE}.log"
                fi
            else
                echo "Executable not found: $EXECUTABLE"
            fi
        done
    done

done