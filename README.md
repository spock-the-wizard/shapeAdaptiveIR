# Shape-adaptive Inverse Rendering 

Implementation of the paper ["Inverse Rendering of Translucent Objects using Shape-adaptive Importance Sampling"](https://diglib.eg.org/items/f57860ef-ec70-44ec-8b02-eddc36dff67b). 
Accepted to Pacific Graphics 2024 Conference Track.

[Project Page](https://spock-the-wizard.github.io/shape-adaptive-IR/)


This is a differentiable renderer for reconstructing scattering parameters of translucent objects based on a [neural BSSRDF model](https://rgl.epfl.ch/publications/Vicini2019Learned).


## Installation
Our implementation is based on the path-space differentiable renderer [PSDR-CUDA](https://diglib.eg.org/items/f57860ef-ec70-44ec-8b02-eddc36dff67b) and the [extension to BSSRDFs](https://github.com/joyDeng/InverseTranslucent). 

To run our code, you can set up the environment yourself by following the instructions found [here](https://psdr-cuda.readthedocs.io/en/latest/core_compile.html).

We also provide a docker container with necessary libraries installed. (Some may still require manual installation, e.g. OptiX)
```bash
docker pull spockthewizard/shapeadaptiveir:latest
```

This code was tested on Ubuntu 20.04.6 LTS.

## Build

```bash
mkdir build
cd build
../cmake.sh # A script for running cmake and make
cd .. && source setpath.sh # Add to PYTHONPATH
```

## Folder Structure

```python
.
├── src/bsdf
│   ├── vaesub.cpp
|   |   # code for shape-adaptive BSSRDF model
│   └── scattereigen.h # helper code 
├── variables # model weights
├── data_stats.json # metadata for running neural model
├── data_kiwi_soap # data (imgs, lights, obj)
│   ├── imgs
│   ├── obj
│   └── light
├── examples/python/scripts # experiment code
```
The provided weights are trained with a more lightweight architecture than proposed in the original forward model paper. No significant performance degradation was noted in our experiments.

## Running Experiments
1. Prepare your data
Put your images, lights and obj file in `/data_kiwi_soap`

2. Set necessary constants (e.g. your path) in `/examples/python/constants.py`

3. Run the following code
```bash
cd examples/python/scripts
./exp_ours.sh ${SCENE_NAME}
```

## Dataset
We provide an item from our synthetic dataset [here](https://drive.google.com/drive/folders/1Jqq-iCiDrXgQrx9BLW3dcvurKiWTD1VO?usp=drive_link).




