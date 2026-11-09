# High-fidelity volumetric imaging of organelle dynamics in living mammals

This is the repository of reconstrcution methods for PICO, with paper entitled "High-fidelity volumetric imaging of organelle dynamics in living mammals".

# System Requirements
We recommend the following configurations:

* 32GB RAM or more
* NVIDIA GPU (supports CUDA) with more than 48GB VRAM, supported by PyTorch

This repository is tested on Ubuntun with a NVIDIA Geforce RTX 4090 (48GB RAM) and 256GB RAM.

# Environment

There is no restriction on operation systems or specific python versions (as long as they are supported by Pytorch). This repository is tested on Python 3.12 with CUDA 12.8 and torch 2.5.1.

# Installation
Clone this repository and create a virtual environment, then install all dependencies as follow:
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install it from the official PyTorch index instead, e.g.:
```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```
See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) to pick the build matching your CUDA version.

# Testing

With the virtual environment activated, run the reconstruction script from the repository root:
```
python test_serenet.py
```

The script reads all `*.tif` files from `./wdf/` and reconsturct them. The full volume will be saved as TIFF format in directory `./wdf/recon_result/`, and the maximum projection will be saved in `./wdf/recon_result/recon_result_mip/`.

