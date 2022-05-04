# Tight-Inclusion Continuous Collision Detection on GPU

A narrow phase CCD algorithm runs on GPU.

This repository is a part of the project [CCD-GPU](https://github.com/dbelgrod/CCD-GPU), where we solve both broad phase and narrow phase of CCD on GPU, using CUDA.

The whole project employs a very simple, yet easily paralleled method for broad phase CCD to pick vertex-triangle pairs or edge-edge pairs which may have collisions. Then, pass the data to narrow phase CCD to solve an accurate time of impact.

