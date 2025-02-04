MATRIX_DIR = $(CURDIR)/../../../out/matrices
BIN_ADD = $(BIN_DIR)/add
RUN_SCRIPT = run_add.sh
RUN_SAME_SIZE_SCRIPT = run_same_size.sh

.PHONY: all clean run run_same_size

all: $(BIN) $(RUN_SCRIPT) $(RUN_SAME_SIZE_SCRIPT)

# Génération du script d'exécution normal
$(RUN_SCRIPT):
	@echo "#!/bin/bash" > $(RUN_SCRIPT)
	@echo "" >> $(RUN_SCRIPT)
	@echo "echo 'Début des opérations de somme de matrices...'" >> $(RUN_SCRIPT)
	@for type in int float; do \
		for file in $(MATRIX_DIR)/$$type/*-number1.csv; do \
			base_name=$$(basename $$file -number1.csv); \
			echo "$(BIN_ADD) $(MATRIX_DIR)/$$type/$$base_name-number1.csv $(MATRIX_DIR)/$$type/$$base_name-number2.csv" >> $(RUN_SCRIPT); \
		done \
	done
	@chmod +x $(RUN_SCRIPT)

# Génération du script pour additionner uniquement les matrices de même taille
$(RUN_SAME_SIZE_SCRIPT):
	@echo "#!/bin/bash" > $(RUN_SAME_SIZE_SCRIPT)
	@echo "" >> $(RUN_SAME_SIZE_SCRIPT)
	@echo "echo 'Addition des matrices de même taille...'" >> $(RUN_SAME_SIZE_SCRIPT)
	@for type in int float; do \
		for size in $$(ls $(MATRIX_DIR)/$$type | grep -oE '^[0-9]+x[0-9]+' | sort -u); do \
			if [ -f "$(MATRIX_DIR)/$$type/$$size-number1.csv" ] && [ -f "$(MATRIX_DIR)/$$type/$$size-number2.csv" ]; then \
				echo "$(BIN_ADD) $(MATRIX_DIR)/$$type/$$size-number1.csv $(MATRIX_DIR)/$$type/$$size-number2.csv" >> $(RUN_SAME_SIZE_SCRIPT); \
			fi \
		done \
	done
	@chmod +x $(RUN_SAME_SIZE_SCRIPT)

run: $(RUN_SCRIPT)
	./$(RUN_SCRIPT)

run_same_size: $(RUN_SAME_SIZE_SCRIPT)
	./$(RUN_SAME_SIZE_SCRIPT)

clean:
	rm -rf $(BIN_DIR) $(RUN_SCRIPT) $(RUN_SAME_SIZE_SCRIPT)
