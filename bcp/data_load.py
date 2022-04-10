# -*- coding: utf-8 -*-

### basic modules
import numpy as np
import time, pickle, os, sys, json, PIL, tempfile, warnings, importlib, math, copy, shutil, setproctitle

### torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import torch.nn.functional as F
from torch import autograd

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np 
from art.utils import to_categorical

class PytorchDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).max(1)[1]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def data_loaders(data, batch_size, augmentation=False, normalization=False, drop_last=False, shuffle=True): #############

    if not os.path.exists('data/'):
        os.makedirs('data/')

    if data == 'cifar10': 
        if augmentation and normalization:
            trainset_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.225, 0.225, 0.225))])
            trainset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=trainset_transforms)
            testset = datasets.CIFAR10(root='./data/', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.225, 0.225, 0.225))]))
            
        elif augmentation and not(normalization):
            trainset_transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
            trainset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=trainset_transforms)
            testset = datasets.CIFAR10(root='./data/', train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor()]))
            
        else:
            trainset = list(datasets.CIFAR10(root='./data/', train=True, download=True, transform=transforms.ToTensor()))[:2500]
            testset = list(datasets.CIFAR10(root='./data/', train=True, download=True, transform=transforms.ToTensor()))[2500:5000]

    if data == "p100":
        x = []
        y = []

        with open("../data/dataset_purchase", "r") as f:
            dataset = f.readlines()
            for datapoint in dataset: 
                split = datapoint.rstrip().split(",")
                label = int(split[0]) - 1
                x.append([int(s) for s in split[1:]])
                y.append(label)

        x = np.array(x).astype(np.float32)
        y = to_categorical(np.array(y), 100)

        target_train_size = 1000
        target_test_size = 1000
        x_target_train = x[:target_train_size]
        y_target_train = y[:target_train_size]
        x_target_test = x[target_train_size:target_train_size+target_test_size]
        y_target_test = y[target_train_size:target_train_size+target_test_size]

        trainset = PytorchDataset(x_target_train, y_target_train)
        testset = PytorchDataset(x_target_test, y_target_test)
        
    if data == 'mnist':
        trainset = datasets.MNIST(root='./data/', train=True, download=True, transform=transforms.ToTensor())
        testset = datasets.MNIST(root='./data/', train=False, download=True, transform=transforms.ToTensor())
    
    if data == 'tinyimagenet':
        if augmentation and normalization:
            trainset_transforms = transforms.Compose([
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            trainset = datasets.ImageFolder(root='../../Imagenet/tiny-imagenet-200/train', transform=trainset_transforms)#, train=True, download=True
            testset = datasets.ImageFolder(root='../../Imagenet/tiny-imagenet-200/val', transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]))
            
        elif not augmentation and normalization:
            trainset_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
            testset_transforms = transforms.Compose([
                    transforms.ToTensor()])
            trainset = datasets.ImageFolder(root='../../Imagenet/tiny-imagenet-200/train', transform=trainset_transforms)#, train=True, download=True
            testset = datasets.ImageFolder(root='../../Imagenet/tiny-imagenet-200/val', transform=testset_transforms)
            
        elif augmentation and not normalization: ##############################
            trainset_transforms = transforms.Compose([
                    transforms.RandomCrop(64, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            testset_transforms = transforms.Compose([
                    transforms.ToTensor()])
            trainset = datasets.ImageFolder(root='../../Imagenet/tiny-imagenet-200/train', transform=trainset_transforms)#, train=True, download=True
            testset = datasets.ImageFolder(root='../../Imagenet/tiny-imagenet-200/val', transform=testset_transforms)
            
        elif not augmentation and not normalization:
            trainset_transforms = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()])
            testset_transforms = transforms.Compose([
                    transforms.ToTensor()])
            trainset = datasets.ImageFolder(root='../../Imagenet/tiny-imagenet-200/train', transform=trainset_transforms)#, train=True, download=True
            testset = datasets.ImageFolder(root='../../Imagenet/tiny-imagenet-200/val', transform=testset_transforms)
        
        
#         testset.targets = list(np.array(testset.targets)-1) ########################
        testset.target_transform = lambda id: id-1



    data_loader  = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=batch_size,
                                              shuffle=shuffle, 
                                              drop_last = drop_last,
                                              num_workers=1)
    test_data_loader  = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=1)
    
    return data_loader, test_data_loader
