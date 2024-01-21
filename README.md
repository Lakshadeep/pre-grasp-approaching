# Pre-grasp approaching

# To run
Go to `/home/sdur/.local/share/ov/pkg/isaac_sim-2022.2.0` and run the following command
```
./python.sh ~/Planning/Codes/pre-grasp-approaching/{script name}.py 

```

# Notes

## Installing new packages in Isaac
```
./python.sh -m pip install {name of the package}  --global-option=build_ext --global-option=build_ext  --global-option="-I/home/sdur/.local/share/ov/pkg/isaac_sim-2022.1.1/kit/python/include/python3.7m"
```

#### Installing this package in Isaac
```
./python.sh -m pip install -e ~/Planning/Codes/base_pose_optimization/  --global-option=build_ext --global-option=build_ext  --global-option="-I/home/sdur/.local/share/ov/pkg/isaac_sim-2022.1.1/kit/python/include/python3.7m"
```

#### Dependencies
omegaconf, hydra, hydra-core, tqdm, opencv-python, networkx, karateclub, mushroom-rl (local), ikfastpy (local), shapely, torch-scatter (build),  torch-sparse (build), torch-cluster (build), torch-spline-conv (build), torch-geometric (build)

Use Isaac 2.0.0 for this package on FacilityCobot workstation.
https://github.com/mberr/pytorch-geometric-installer