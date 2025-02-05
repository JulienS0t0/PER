# PER

# Installation de l'environement :

## Drivers du GPU
```
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
```

## Cuda toolkit
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

## Opencl (nécessite les drivers du GPU, ici nvidia)
```
sudo apt install nvidia-opencl-dev ocl-icd-libopencl1 opencl-headers clinfo
```

# Build
```
make
make gen
```

# Run
## Scripts & Makefiles
Depuis le dossier racine du projet
```
make run add
```
Remplacer add par toute autre opération

## Manuellement

### CPU

Exemple :
```
./out/operations/cuda/add ./out/matrices/int/4096x4096-number1.csv ./out/matrices/int/4096x4096-number2.csv
```

### Cuda

Exemple :
```
nvprof ./out/operations/cuda/add ./out/matrices/int/4096x4096-number1.csv ./out/matrices/int/4096x4096-number2.csv
```

### OpenCL

Exemple :
```
nsys profile ./out/operations/opencl/add ./out/matrices/int/4096x4096-number1.csv ./out/matrices/int/4096x4096-number2.csv

nsys-ui
```
sélectionner le fichier report.nsys-rep pour l'inspecter