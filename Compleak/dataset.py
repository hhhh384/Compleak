import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import TensorDataset

def get_imagenet(name, train=True):
    print(f"Build Dataset {name}")
    if name == "mini_imagenet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        else:
            transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = torchvision.datasets.ImageFolder("../data/mini_imagenet/train",transform=transform) 
        test_dataset = torchvision.datasets.ImageFolder("../data/mini_imagenet/test",transform=transform)
        total_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    return total_dataset 
    


def get_dataset(name):
    print(f"Build Dataset {name}")
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        trainset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=True
                                                , download=True,transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data/datasets/cifar10-data', train=False
                                                , download=True,transform=transform)

    elif name == "cifar100":
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data/datasets/cifar100-data', train=False, download=True, transform=transform)
   

    return trainset, testset
