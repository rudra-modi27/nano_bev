# NanoBEV: Real-Time 3D Spatial Perception on a 4GB VRAM Budget

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![nuScenes](https://img.shields.io/badge/Dataset-nuScenes-blue?style=for-the-badge)

A highly optimized, multi-camera Bird's Eye View (BEV) occupancy network engineered specifically for heavily constrained Edge Computing hardware. 

While modern autonomous driving architectures (like Tesla's FSD or comma.ai) require massive server-grade GPUs to process 3D spatial geometry, **this project successfully fuses 6 high-resolution camera feeds into a unified 400x400 map at 12 FPS on a consumer-grade 4GB RTX 3050.**

## The Engineering Philosophy & Architecture

This repository contains two distinct architectural approaches, documenting the journey from a highly precise 2D-projection model to a bleeding-edge 3D Knowledge Distillation experiment.

### Architecture 1: The "High-Precision" Production Pipeline
The primary challenge of edge-deployed AI is dealing with severe class imbalance (98% of a BEV grid is empty road, 2% is vehicles) without the VRAM to support massive batch sizes.

* **Vision Backbone:** A custom-frozen `ConvNeXt-Tiny` processes the 6 synchronized camera feeds.
* **Spatial Projection:** Uses a VRAM-optimized multi-view feature squish to collapse 2D features into a 3D grid.
* **Loss Optimization:** Implemented a **Distance-Weighted Focal + Soft Dice Loss**. By mathematically dropping the penalty for "easy" background pixels, the network was forced to hyper-focus on vehicle geometry.
* **The Result:** Reached **~70% Precision** on entirely unseen validation streets using only 8 scenes of training data. When the model draws a bounding box, it is extremely reliable.

### Architecture 2: The "Lift-Splat-Shoot" (LSS) Distillation Experiment
To push the 4GB hardware to its absolute mathematical limit, I engineered an implicit LSS pipeline to give the model true 3D depth perception.

* **The Problem:** 4GB of VRAM is not enough to perform "Explicit Splatting" (multiplying the 3D frustums by the Camera Extrinsic matrices for all 6 cameras simultaneously).
* **The Hack:** Implemented an "Implicit Splat" combined with **Knowledge Distillation**. 
* **The Teacher:** I dynamically piped the images through `MiDaS_small` (frozen as a zero-grad encyclopedia) to generate perfect depth maps, mathematically forcing the blind BEV model to learn 3D distance via L1 Distillation Loss.
* **The Result:** The model successfully learned 3D depth perception at **12 FPS**, but proved that without the VRAM for Explicit matrix splatting, the 3D frustums "smear" across the grid. This defines the exact boundary of what is possible on 4GB hardware.

## Performance Metrics (Hardware: RTX 3050 - 4GB)

| Metric | Architecture 1 (Focal/2D) | Architecture 2 (LSS/Distilled) |
| :--- | :--- | :--- |
| **Inference Speed** | `10.28 FPS` | `11.99 FPS` |
| **Validation Precision**| `69.95%` | `36.23%` |
| **Grid Resolution** | `400 x 400` | `400 x 400` |
| **VRAM Usage** | `~3.6 GB` | `~3.9 GB` |

*Note: Models were strictly trained on `nuScenes-mini` to enforce a low-data constraint. Recall is bottlenecked by the 8-scene training limit.*

## Quick Start

**1. Clone & Install**
```bash
git clone [https://github.com/yourusername/NanoBEV.git](https://github.com/yourusername/NanoBEV.git)
cd NanoBEV
pip install torch torchvision pyquaternion nuscenes-devkit opencv-python timm
