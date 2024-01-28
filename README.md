# Pre-grasp approaching

Code repository for our paper "Pre-grasp approaching on mobile robots: a pre-active layered approach" by Lakshadeep Naik, Sinan Kalkan and Norbert Kruger. 


# Pre-requisites
Our code uses NVIDIA Isaac Sim for simulation. Installation instructions can be found [here](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html). This code has been tested with Isaac Sim version 'isaac_sim-2022.2.0'

Further, following python packages should be installed in the Isaac sim python environment:
```
omegaconf, hydra, hydra-core, tqdm, opencv-python, networkx, karateclub, mushroom-rl (local), shapely, torch-scatter (build),  torch-sparse (build), torch-cluster (build), torch-spline-conv (build), torch-geometric (build)
```

'local' - local installation of the package is required
'build' - depending on the CUDA version and other driver installations on your system you might need to build the package locally.

#### Installing new python packages in Isaac
```
./python.sh -m pip install {name of the package}  --global-option=build_ext --global-option=build_ext  --global-option="-I{Isaac install path}/ov/pkg/isaac_sim-2022.2.0/kit/python/include/python3.7m"
```

#### Installing local python package in Isaac (for mushoorm-rl and this package)
```
./python.sh -m pip install -e {package path}/  --global-option=build_ext --global-option=build_ext  --global-option="-I{Isaac install pathj}/ov/pkg/isaac_sim-2022.2.0/kit/python/include/python3.7m"
```

## NOTES:
- Before trying to run the code, please change relative paths in all the config files in 'conf' folder.

# To run the scripts
First open `{Isaac install path}/ov/pkg/isaac_sim-2022.2.0` in terminal and run the following command:
```
./python.sh {package path}/{script name}.py 

```

# Training
Layer 1: base motion
```
./python.sh {package path}/pre-grasp-approaching/train/base_motion.py 

```