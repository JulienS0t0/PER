# PER

# Installation de l'environement :

## Drivers du GPU
'''
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot
'''

## Cuda toolkit
'''
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
'''

## Opencl (nécessite les drivers du GPU, ici nvidia)
'''
sudo apt install nvidia-opencl-dev ocl-icd-libopencl1 opencl-headers clinfo
'''
