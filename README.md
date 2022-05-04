# Narrow Phase Continuous Collision Detection on GPU

A narrow phase CCD algorithm runs on GPU.

This repository is a part of the project [CCD-GPU](https://github.com/dbelgrod/CCD-GPU), where we solve both broad phase and narrow phase of CCD on GPU, using CUDA.

The whole project employs a very simple, yet easily paralleled method for broad phase CCD to pick vertex-triangle pairs or edge-edge pairs which may have collisions. Then, pass the data to narrow phase CCD to solve an accurate time of impact. The narrow phase follows the idea of [Tight-Inclusion](https://github.com/Continuous-Collision-Detection/Tight-Inclusion), but did a lot of changes to make sure it can be executed on GPU, efficiently with guaranteed accuracy. 

For broad phase CCD, please refer to [Broad-Phase CCD](https://github.com/dbelgrod/broadphase-gpu). For narrow phase CCD, please refer to this repository.

