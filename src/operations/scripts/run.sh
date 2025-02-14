#!/usr/bin/env bash

# Usage : run.sh <operation> [save]
# This file should be called from the root of the project
# Please use it via the make file 'make run operation' or 'make run operation save'

if [ -z "$1" ]; then
    echo "Usage: $0 <operation> [save]"
    exit 1
fi

# Obtenir la date et l'heure actuelles
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

OPERATION=$1
SAVE_ENABLED=false

if [ "$2" == "save" ]; then
    SAVE_ENABLED=true
fi

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

# Création des dossiers pour les résultats et les logs
for DIR in "$CUDA_RESULTS_DIR" "$OPENCL_RESULTS_DIR" "$CPU_RESULTS_DIR" "$CPU_OPTI_O2_RESULTS_DIR" "$CPU_OPTI_O3_RESULTS_DIR"; do
    mkdir -p "$DIR/log"
done

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
        for IMPL in cpu cpu_opti_O2 cpu_opti_O3 cuda opencl; do
            EXECUTABLE="$OPERATIONS_DIR/$IMPL/$OPERATION"
            RESULT_DIR=""

            case "$IMPL" in
                cuda) RESULT_DIR="$CUDA_RESULTS_DIR" ;;
                opencl) RESULT_DIR="$OPENCL_RESULTS_DIR" ;;
                cpu_opti_O2) RESULT_DIR="$CPU_OPTI_O2_RESULTS_DIR" ;;
                cpu_opti_O3) RESULT_DIR="$CPU_OPTI_O3_RESULTS_DIR" ;;
                cpu) RESULT_DIR="$CPU_RESULTS_DIR" ;;
            esac

            LOG_FILE="${RESULT_DIR}/log/${MATRIX_SIZE}_${TYPE}.log"
            OUTPUT_FILE="${RESULT_DIR}/${MATRIX_SIZE}_${TYPE}.csv"

            SAVE_ARG=""
            if [ "$SAVE_ENABLED" = true ]; then
                SAVE_ARG="$OUTPUT_FILE"
            fi

            if [ -x "$EXECUTABLE" ]; then
                echo "Running $EXECUTABLE with $FILE1 and $FILE2"
                $TIME_CMD -v "$EXECUTABLE" "$FILE1" "$FILE2" $SAVE_ARG &> "$LOG_FILE"
            else
                echo "Executable not found: $EXECUTABLE"
            fi
        done
    done
done
