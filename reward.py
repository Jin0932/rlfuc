##!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from typing import List, Tuple
import torchvision.transforms.functional as TFs
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from pretraining import Net
from utils import * 
from generate_dataset import get_test_FashionMNIST_imb_data, get_test_FashionMNIST_imb_data

from tensorboardX import SummaryWriter

class RewardNet(nn.Module):
    def __init__(self, pretrain_net):

        super(RewardNet, self).__init__()
        self.pretrain_net = pretrain_net
        self.features = nn.Sequential(*list(self.pretrain_net.network.children())[:-1])
        for param in self.features.parameters():
            param.requires_grad = False
        last_layer = list(self.pretrain_net.network.children())[-1]
        num_features = last_layer.in_features
        self.reward_model = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, image):
        features_output = self.features(image.to(device))
        return self.reward_model(features_output.to(device))

def show_image(image_tensor):
        image = image_tensor.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.show()

def train_reward_function(dataset,  pretrain_name, reward_name, batch_size) -> RewardNet:
    writer = SummaryWriter(REWARD_LOG_DIR + reward_name +"/")
    pretraining_model = Net().to(device)
    pretraining_model.load_state_dict(torch.load(PRETRAING_MODEL_DIR + pretrain_name+".pth"))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    reward_net = RewardNet(pretraining_model).to(device)
    optimizer = torch.optim.Adam(reward_net.parameters(),lr=1e-3)
    for epoch in tqdm(range(80)):
        total_loss = 0.0
        total_samples = 0
        reward_net.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss =F.cross_entropy(reward_net(inputs), labels)
            loss.backward()
            optimizer.step()
        writer.add_scalar('reward training loss', loss, epoch)
    torch.save(reward_net.state_dict(), REWARD_MODEL_DIR + reward_name+".pth")


def test_reward_function(dataset,pretrain_name, reward_name, batch_size):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    pretraining_model = Net().to(device)
    pretraining_model.load_state_dict(torch.load(PRETRAING_MODEL_DIR + pretrain_name+".pth"))
    
    reward_net = RewardNet(pretraining_model).to(device)
    reward_net.load_state_dict(torch.load(REWARD_MODEL_DIR + reward_name+".pth"))
    for image, label in test_loader:
        with torch.no_grad():
            image = image.to(device)
            label = label.to(device)        
            logit = reward_net(image)
            reward = F.cross_entropy(logit, label)
            for i in range(10):
                aa = torch.tensor([i], dtype=torch.int64).to(device)
     