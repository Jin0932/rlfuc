##!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from utils import * 
from pretraining import * 
from reward import *
from finetune_classification import *
from Env import *
import gc

from generate_dataset import *
names = ["101"] 
values = [1.0, 0.46294,0.21431,0.09921,0.04593,0.02126,0.00984,0.00456,0.00211,0.00098]
all_values = [values]
count = 1
for (name, value) in zip(names, all_values):
    pretrain_name =name
    reward_name=name
    rl_name = name
    gc.collect()

    train_data, val_data = get_train_FashionMNIST_imb_data(value)
    test_data = get_test_FashionMNIST_imb_data(value)
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    train_mnist_pretrain(device, train_data, net, criterion, optimizer, pretrain_name)
    balance_acc = test_mnist_pretrain(device, test_data, net, pretrain_name)
    train_reward_function(train_data.dataset, pretrain_name, reward_name, batch_size=64)
    args = parse_args()
    envs = ClassificationEnv(train_data, pretrain_name, reward_name)
    agent = Agent(envs, pretrain_name).to(device)
    train_finetune(args, envs, agent, pretrain_name, reward_name, rl_name, val_data, balance_acc)
    evaluate_finetune(envs, agent, test_data, rl_name)



