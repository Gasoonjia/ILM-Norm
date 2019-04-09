# Instance-Level Meta Normalization (ILM-Norm)
This repository contains a [PyTorch](http://pytorch.org/) implementation of the paper [Instance-Level Meta Normalization](https://arxiv.org/abs/1904.03516) presented at CVPR 2019. 

The code is based on the [PyTorch example for training ResNet on Imagenet](https://github.com/pytorch/examples/tree/master/imagenet) and [Train CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar).

## Table of Contents
0. [Introduction](#introduction)
0. [Usage](#usage)
0. [Requirements](#requirements)
0. [Citing](#citing)

## Introduction
This paper presents a normalization mechanism called Instance-Level Meta Normalization (ILM-Norm) to address a learning-to-normalize problem. ILM-Norm learns to predict the normalization parameters via both the feature feed-forward and the gradient back-propagation paths.
ILM-Norm provides a meta normalization mechanism and has several good properties. It can be easily plugged into existing instance-level normalization schemes such as Instance Normalization, Layer Normalization, or Group Normalization. ILM-Norm normalizes each instance individually and therefore maintains high performance even when small mini-batch is used. The experimental results show that ILM-Norm well adapts to different network architectures and tasks, and it consistently improves the performance of the original models.

## Usage
There are two training files. One for CIFAR-10 and Cifar-100 `train.py` and the other for ImageNet `imageNet.py`.

### Cifar
The network can be simply trained with `python train.py` or with optional arguments for different hyperparameters:
```sh
python train.py --data [cifar-10/cifar-100 folder]
```

The network can be also simply infered with the following command:
```sh
python train.py --infer [checkpoint folder] --data [cifar-10/cifar-100 folder]
```

### ImageNet
For ImageNet the folder containing the dataset needs to be supplied

```sh
python imageNet.py --data [imagenet folder]
```

You can also infer the network with the following command:

```sh
python imageNet.py --infer [checkpoint folder] --data [imagenet folder]
```

## Requirements 
This implementation is developed for 

0. Python 3.6.5
0. PyTorch 1.0.1
0. CUDA 9.1

For compatibility to newer versions, please make a pull request.

## Citing
If you find this helps your research, please consider citing:

```
@conference{Jia2019,
title = {Instance-Level Meta Normalization},
author = {Songhao Jia and Ding-Jie Chen and Hwann-Tzong Chen},
year = {2019},
journal = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
}
```
```
