{ pkgs ? import <nixpkgs> { config = { allowUnfree = true; }; } }:

pkgs.mkShell {
  buildInputs = [
    pkgs.libgcc
    pkgs.opencl-headers
    pkgs.ocl-icd
    pkgs.cudatoolkit
    pkgs.cudatoolkit.lib
    pkgs.cudaPackages.cuda_cudart
  ];

  shellHook = ''
    # Set environment variables for CUDA
    export CUDA_PATH=${pkgs.cudatoolkit}
    export PATH=$CUDA_PATH/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:$LD_LIBRARY_PATH
    export CPATH=$CUDA_PATH/include:$CPATH
    export LIBRARY_PATH=$CUDA_PATH/lib64:$LIBRARY_PATH
  '';
}

