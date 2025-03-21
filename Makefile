# Chemins vers les autres Makefiles
SUBDIRS = src/matrices src/operations/cpu src/operations/cuda src/operations/opencl src/operations/cpu_opti

# Emplacement des binaires
BINARIES = out/matrices out/operations/cpu out/operations/cuda out/operations/opencl out/operations/cpu_opti

# Matrices
MATRICES = out/matrices/float/*.csv out/matrices/int/*.csv

# Commandes principales
.PHONY: all build gen run run-save clean clean-res

all: build # gen

build:
	@for dir in $(SUBDIRS); do $(MAKE) -C $$dir; done
	
gen:
	@$(MAKE) -C src/matrices gen

run: 
	@./src/operations/scripts/run.sh $(word 2, $(MAKECMDGOALS))

run-save:
	@./src/operations/scripts/run.sh $(word 2, $(MAKECMDGOALS)) save

result:
	@python3 ./src/operations/scripts/compute-res.py

clean:
	@for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean; done
	rm -rf out/operations/* out/matrices/*

clean-res:
	rm -rf res/*
	# @for dir in $(SUBDIRS); do $(MAKE) -C $$dir clean-res; done