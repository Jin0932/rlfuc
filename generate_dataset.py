##!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset
import random

def get_train_FashionMNIST_imb_data(class_distribution):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])  
    fashion_mnist_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    train_size = int(0.9 * len(fashion_mnist_dataset)) 
    val_size = len(fashion_mnist_dataset) - train_size 
    train_dataset, val_dataset = torch.utils.data.random_split(fashion_mnist_dataset, [train_size, val_size])

    class_train_subsets = [Subset(train_dataset, [idx for idx in range(len(train_dataset)) if train_dataset[idx][1] == digit]) for digit in range(10)]
    class_val_subsets = [Subset(val_dataset, [idx for idx in range(len(val_dataset)) if val_dataset[idx][1] == digit]) for digit in range(10)]

    imbalanced_train_datasets = []
    imbalanced_val_datasets = []

    for digit, subset in enumerate(class_train_subsets):
        num_samples = int(len(subset) * class_distribution[digit])
        imbalanced_subset = Subset(subset, torch.randperm(len(subset))[:num_samples])
        imbalanced_train_datasets.append(imbalanced_subset)
    for digit, subset in enumerate(class_val_subsets):
        num_samples = int(len(subset) * class_distribution[digit])
        imbalanced_subset = Subset(subset, torch.randperm(len(subset))[:num_samples])
        imbalanced_val_datasets.append(imbalanced_subset)
        
    imbalanced_train_dataset = ConcatDataset(imbalanced_train_datasets)
    imbalanced_val_dataset = ConcatDataset(imbalanced_val_datasets)
    batch_size = 64
    imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=batch_size, shuffle=True)
    imbalanced_val_loader = torch.utils.data.DataLoader(imbalanced_val_dataset, batch_size=batch_size, shuffle=False)
    return imbalanced_train_loader, imbalanced_val_loader


def get_test_FashionMNIST_imb_data(class_distribution):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    fashion_mnist_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    class_subsets = [Subset(fashion_mnist_dataset, [idx for idx in range(len(fashion_mnist_dataset)) if fashion_mnist_dataset.targets[idx] == digit]) for digit in range(10)]

    imbalanced_datasets = []
    for digit, subset in enumerate(class_subsets):
        num_samples = int(len(subset) * class_distribution[digit])
        imbalanced_subset = Subset(subset, torch.randperm(len(subset))[:num_samples])
        imbalanced_datasets.append(imbalanced_subset)
    imbalanced_dataset = ConcatDataset(imbalanced_datasets)
    batch_size = 64
    imbalanced_dataloader = torch.utils.data.DataLoader(imbalanced_dataset, batch_size=batch_size, shuffle=True)
    return imbalanced_dataloader