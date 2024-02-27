##!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import time
from distutils.util import strtobool
from matplotlib import pyplot as plt
from torchsummary import summary
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

from Env import ClassificationEnv
from pretraining import Net
from utils import *

from metrics import balanced_accuracy, plot_normalized_confusion_mtx, write_true_prediction
 


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=False`")
    
    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=800000, help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="the learning rate of the optimizer")   # 1e-5,2e-4
    parser.add_argument("--num-envs", type=int, default=1, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128, help="the number of steps to run in each environment per policy rollout")   # 200
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    parser.add_argument("--kl-ctl-value", type=float, default=0.8, help="Pretraining and RL KL divergence threshold")
    
    parser.add_argument("--initial_kl_coefficient", type=float, default=0.6, help="Pretraining and RL KL divergence threshold")
    parser.add_argument("--final_kl_coefficient", type=float, default=0.2, help="Pretraining and RL KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

class Agent(nn.Module):
    def __init__(self, envs, pretrain_name):
        super().__init__()
        self.pretrain_name = pretrain_name
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

    def get_pretraining_model(selfs):
        pretraining_model = Net().to(device)
        pretraining_model.load_state_dict(torch.load(PRETRAING_MODEL_DIR + selfs.pretrain_name+".pth"))
        return pretraining_model

    def critic(self, input):
        features = nn.Sequential(*list(self.get_pretraining_model().network.children())[:-1])
        new_critic_network = nn.Sequential(features, nn.Linear(512, 1)).to(device)
        return new_critic_network(input)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.network(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)


def train_finetune(args, envs, agent, pretrain_name, reward_name, rl_name, val_data, balance_acc):
    writer = SummaryWriter(RL_LOG_DIR + rl_name + "/")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
   
    pretraining_model =agent.get_pretraining_model()
    pretraining_model.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    observation_space = (1, 28, 28)
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    total_steps = num_updates
    decay_rate = (args.initial_kl_coefficient -args.final_kl_coefficient) / total_steps
    args.kl_ctl_value = args.initial_kl_coefficient
    
    best_balance_acc = 0
    best_epoch = 0
    for update in tqdm(range(1, num_updates + 1), desc="PPO微调模型--Episode训练进度"):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.to(device))
                values[step] = value.flatten()

            with torch.no_grad():
                ref_logits = pretraining_model(next_obs)
                ref_probs = Categorical(logits=ref_logits)
                action_pre = ref_probs.sample()
                ref_logprob = ref_probs.log_prob(action_pre)
                
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, done = envs.step(action)
            
            kl = logprob - ref_logprob 
            non_score_reward = -args.kl_ctl_value * kl
            reward = 0.1*reward + non_score_reward
            rewards[step] = reward.clone().detach().to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)
           
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        b_obs = obs.reshape((-1,) + (1,28,28))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + ())
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                # $L^{C L I P}(\theta)=\hat{\mathbb{E}}_t\left[\min \left(r_t(\theta) \hat{A}_t, \operatorname{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \hat{A}_t\right)\right]$
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                # $L_t^{C L I P+V F+S}(\theta)=\hat{\mathbb{E}}_t\left[L_t^{C L I P}(\theta)-c_1 L_t^{V F}(\theta)+c_2 S\left[\pi_\theta\right]\left(s_t\right)\right]$
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    
def evaluate_finetune(envs, agent, test_data, rl_name):
    agent.load_state_dict(torch.load(FINETUNE_RL_MODEL_DIR + rl_name+".pth"))
    y_true, y_predicted= [], []
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_data:
            images, labels = data
            for image, label in zip(images, labels):
                # print(image.shape)
                action, logprob, _, value = agent.get_action_and_value(torch.Tensor(image).unsqueeze(0).to(device))
                y_true.append(label.item())
                y_predicted.append(action.item())
                correct += (action == label)
                total += 1
        balance_acc, recall, precision, F1 = balanced_accuracy(y_true, y_predicted, RL_ACC_DIR + rl_name+".txt")
        print("【 PPO微调 】balance_acc={:.2f}%, recall={:.2f}%, precision={:.2f}%, F1={:.2f}%".format(balance_acc*100, recall*100, precision*100, F1*100))
        plot_normalized_confusion_mtx(y_true, y_predicted, RL_CONFUSION_MATRIX_DIR+ rl_name + ".pdf")
        write_true_prediction(y_true, y_predicted, RL_DATA_DIR + rl_name+".txt")
    
