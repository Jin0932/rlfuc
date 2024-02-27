##!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import seaborn as sns
import numpy as np
from tqdm import tqdm

from metrics import balanced_accuracy, plot_normalized_confusion_mtx, write_true_prediction
from utils import * 

from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    def forward(self,x):
        return self.network(x)

def train_mnist_pretrain(device, train_loader, net, criterion, optimizer, pretrain_name):
    writer = SummaryWriter(PRETRAINING_LOG_DIR + pretrain_name+"/")
    for epoch in tqdm(range(30)):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        writer.add_scalar('Pretrain training loss', running_loss, epoch)
    torch.save(net.state_dict(), PRETRAING_MODEL_DIR + pretrain_name+".pth")


def test_mnist_pretrain(device, test_loader, net, pretrain_name):
    net.load_state_dict(torch.load(PRETRAING_MODEL_DIR + pretrain_name+".pth"))
    correct = 0
    total = 0
    y_true, y_predicted = [], []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # print("image.shape ", images[0].shape)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) 
            
            # data transfer
            y_true_tensor = [int(x) for x in labels]
            y_predict_tensor = [int(x) for x in predicted]
            y_true = y_true + y_true_tensor
            y_predicted = y_predicted + y_predict_tensor
            
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        balance_acc, recall, precision, F1 = balanced_accuracy(y_true, y_predicted, PRETRAIN_ACC_DIR + pretrain_name+".txt")
        print("【CNN预训练】balance_acc={:.2f}%, recall={:.2f}%, precision={:.2f}%, F1={:.2f}%".format(balance_acc*100, recall*100, precision*100, F1*100))
        
        plot_normalized_confusion_mtx(y_true, y_predicted, PRETRAINING_CONFUSION_MATRIX_DIR + pretrain_name + ".pdf")
        write_true_prediction(y_true, y_predicted, PRETRAIN_DATA_DIR + pretrain_name+".txt")
        return balance_acc
        