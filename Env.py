##!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset

from reward import RewardNet
from pretraining import Net
from utils import *

class ClassificationEnv:
    def __init__(self, train_data, pretrain_name, reward_name):
        self.dataset = train_data.dataset
        self.pretraining_model = Net().to(device)
        self.pretraining_model.load_state_dict(torch.load(PRETRAING_MODEL_DIR + pretrain_name+".pth"))
        self.reward_model_net = RewardNet(self.pretraining_model).to(device)
        self.reward_model_net.load_state_dict(torch.load(REWARD_MODEL_DIR + reward_name+".pth"))
        self.current_index = 0
        self.total_images = len(self.dataset)
        indices = torch.randperm(len(self.dataset))
        self.train_dataset = Subset(self.dataset, indices)
        
    def step(self, action):
        image, label = self.train_dataset[self.current_index]
        self.current_index = (self.current_index + 1) % self.total_images
        with torch.no_grad():
            image, action = image.to(device),action.to(device)
            logit = self.reward_model_net(image.unsqueeze(0))
            reward = -F.cross_entropy(logit, action)
        done = (label == action.item())
        next_image, next_label = self.train_dataset[self.current_index]
        return next_image.unsqueeze(0), float(reward.item()), done

    def reset(self):
        self.current_index = 0
        indices = torch.randperm(len(self.dataset))
        self.train_dataset = Subset(self.dataset, indices)
        state, label = self.train_dataset[self.current_index]
        return state.unsqueeze(0)
    
    def show_image(self, image_tensor):
        image = image_tensor.reshape(28, 28)
        plt.imshow(image, cmap='gray')
        plt.show()