# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from customKing.data import DatasetCatalog

from .Image_classification.Cifar10 import register_Cifar10
from .Image_classification.Cifar100 import register_Cifar100
from .Image_classification.SVHN import register_SVHN
from .Image_classification.BIRDS import register_BIRDS
from .Image_classification.CARS import register_CARS
from .Image_classification.ImageNet import register_ImageNet
from .Image_classification.MNIST import register_MNIST
from .Image_classification.MNIST_Fashion import register_MNIST_Fashion
from .Image_classification.USPS import register_USPS
from .Image_classification.Gaussian_noise import register_Gaussian_Noise
from .Image_classification.Uniform_noise import register_Uniform_Noise
from .Image_classification.Sphere_OOD import register_Sphere_OOD
from .Image_classification.PACS import register_PACS


def register_all_Cifar10(root):
    names = ["Cifar10_train","Cifar10_valid","Cifar10_train_and_valid","Cifar10_test",
            "Cifar10_train_and_valid_and_test","Cifar10_train_aug","Cifar10_valid_aug","Cifar10_train_and_valid_aug",
            "Cifar10_test_aug","Cifar10_train_and_valid_and_test_aug"]
    for name in names:
        register_Cifar10(name,root)

def register_all_Cifar100(root):
    names = ["Cifar100_train","Cifar100_valid","Cifar100_train_and_valid","Cifar100_test",
            "Cifar100_train_and_valid_and_test"]
    for name in names:
        register_Cifar100(name,root)

def register_all_SVHN(root):
    names = ["SVHN_train","SVHN_valid","SVHN_train_and_valid","SVHN_test","SVHN_train_and_valid_and_test"]
    for name in names:
        register_SVHN(name,root)


def register_all_BIRDS(root):
    names = ["BIRDS_train","BIRDS_valid","BIRDS_train_and_valid","BIRDS_test","BIRDS_train_and_valid_and_test"]
    for name in names:
        register_BIRDS(name,root)

def register_all_CARS(root):
    names = ["CARS_train","CARS_valid","CARS_train_and_valid","CARS_test","CARS_train_and_valid_and_test"]
    for name in names:
        register_CARS(name,root)

def register_all_ImageNet(root):
    names = ["ImageNet_train","ImageNet_valid","ImageNet_train_and_valid","ImageNet_test","ImageNet_train_and_valid_and_test"]
    for name in names:
        register_ImageNet(name,root)

def register_all_MNIST(root):
    names = ["MNIST_train","MNIST_valid","MNIST_train_and_valid","MNIST_test","MNIST_train_and_valid_and_test","MNIST_train_before_5","MNIST_valid_before_5",
             "MNIST_train_and_valid_before_5","MNIST_test_before_5","MNIST_Digits_Fashion_test"]
    for name in names:
        register_MNIST(name,root)

def register_all_MNIST_Fashion(root):
    names = ["MNIST_Fashion_train","MNIST_Fashion_valid","MNIST_Fashion_train_and_valid","MNIST_Fashion_test","MNIST_Fashion_train_and_valid_and_test"]
    for name in names:
        register_MNIST_Fashion(name,root)

def register_all_USPS(root):
    names = ["USPS_train","USPS_valid","USPS_train_and_valid","USPS_test","USPS_train_and_valid_and_test"]
    for name in names:
        register_USPS(name,root)

def register_all_Gaussian_Noise():
    names = ["Gaussian_Noise_test"]
    for name in names:
        register_Gaussian_Noise(name)

def register_all_Uniform_Noise():
    names = ["Uniform_Noise_test"]
    for name in names:
        register_Uniform_Noise(name)

def register_all_Sphere_OOD(root):
    names = ["Sphere_test"]
    for name in names:
        register_Sphere_OOD(name,root)

def register_all_PACS(root):
    names = ["PACS_train","PACS_test"]
    for name in names:
        register_PACS(name,root)

if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("CUSTOM_KING_DATASETS", "datasets"))
    #_root = "datasets"
    register_all_Cifar10(_root)
    register_all_Cifar100(_root)
    register_all_SVHN(_root)
    register_all_BIRDS(_root)
    register_all_CARS(_root)
    register_all_ImageNet(_root)
    register_all_MNIST(_root)
    register_all_MNIST_Fashion(_root)
    register_all_USPS(_root)
    register_all_Gaussian_Noise()
    register_all_Uniform_Noise()
    register_all_Sphere_OOD(_root)
    register_all_PACS(_root)