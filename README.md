# GRACE: Graph-Regularized Attentive Convolutional Entanglement with Laplacian Smoothing for Robust DeepFake Video Detection

The official pytorch implementation of "GRACE: Graph-Regularized Attentive Convolutional Entanglement with Laplacian Smoothing for Robust DeepFake Video Detection". Submitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI 2024).

## [[Paper Link (arXiv)]](https://arxiv.org/abs/2404.15781)

[Chih-Chung Hsu](https://cchsu.info/), Shao-Ning Chen, Mei-Hsuan Wu, Yi-Fang Wang, [Chia-Ming Lee](https://ming053l.github.io/), [Yi-Shiuan Chou](https://nelly0421.github.io/)

Advanced Computer Vision LAB, National Cheng Kung University

## Overview

In the field of Deepfake detection, one particular issue lies with facial images being mis-detected, often originating from camera motion, degraded videos or adversarial attacks, leading to unexpected temporal artifacts that can undermine the efficacy of DeepFake video detection techniques.

<img src=".\figures\face_detection.png" width="800"/>

This paper introduces a novel method for robust DeepFake video detection, harnessing the power of the proposed Graph-Regularized Attentive Convolutional Entanglement (GRACE) based on the graph convolutional network with graph Laplacian to address the aforementioned challenges.

<img src=".\figures\GRACE.png" width="800"/>

## Environment

- [PyTorch >= 1.7](https://pytorch.org/)
- CUDA >= 11.2
- python==3.8.18
- pytorch==1.11.0 
- cudatoolkit=11.3 
- onnx==1.14.1
- onnxruntime==1.16.1

### Installation
```
git clone https://github.com/ming053l/GRACE.git
conda create --name grace python=3.8 -y
conda activate grace
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
cd GRACE
pip install -r requirements.txt
```

## How to Train

## How to Test

## Citations

If our work is helpful to your reaearch, please kindly cite our work. Thank!

#### BibTeX


## Contact
If you have any question, please email zuw408421476@gmail.com to discuss with the author.
