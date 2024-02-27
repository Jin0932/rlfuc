##!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch
import os
device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
dataset_name = "fashionmnist"
# Model directory
PRETRAING_MODEL_DIR = "Model/" + dataset_name + "/pretrain_model/"
REWARD_MODEL_DIR = "Model/" + dataset_name + "/reward_model/" 
FINETUNE_RL_MODEL_DIR = "Model/" + dataset_name + "/finetune_model/"
os.makedirs(PRETRAING_MODEL_DIR, exist_ok=True)
os.makedirs(REWARD_MODEL_DIR, exist_ok=True)
os.makedirs(FINETUNE_RL_MODEL_DIR, exist_ok=True)

# Logs directory
PRETRAINING_LOG_DIR = "Runs/" + dataset_name + "-"
REWARD_LOG_DIR ="Runs/" + dataset_name + "-"
RL_LOG_DIR ="Runs/" + dataset_name + "-"
os.makedirs(PRETRAINING_LOG_DIR, exist_ok=True)
os.makedirs(REWARD_LOG_DIR, exist_ok=True)
os.makedirs(RL_LOG_DIR, exist_ok=True)

# Experiment data
PRETRAIN_ACC_DIR = "ResultData/"+dataset_name+"/pretrain_acc/"
RL_ACC_DIR = "ResultData/"+dataset_name+"/rl_finetune_acc/"
os.makedirs(PRETRAIN_ACC_DIR, exist_ok=True)
os.makedirs(RL_ACC_DIR, exist_ok=True)
PRETRAIN_DATA_DIR = "ResultData/"+dataset_name+"/pretrain_data/"
RL_DATA_DIR = "ResultData/"+dataset_name+"/rl_finetune_data/"
os.makedirs(PRETRAIN_DATA_DIR, exist_ok=True)
os.makedirs(RL_DATA_DIR, exist_ok=True)

# confusion_matrix_fashion
PRETRAINING_CONFUSION_MATRIX_DIR = "ResultsPicture/" + dataset_name + "/pretrain_confusion_matrix/"
RL_CONFUSION_MATRIX_DIR = "ResultsPicture/" + dataset_name + "/rl_confusion_matrix/"

os.makedirs(PRETRAINING_CONFUSION_MATRIX_DIR, exist_ok=True)
os.makedirs(RL_CONFUSION_MATRIX_DIR, exist_ok=True)
