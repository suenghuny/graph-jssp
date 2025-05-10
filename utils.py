import math

import torch
import torch.nn as nn

def initialize_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def create_feature_actions(feature_, action_):
    N = feature_.size(0)
    # Flatten sequence of features.
    f = feature_[:, :-1].view(N, -1)
    n_f = feature_[:, 1:].view(N, -1)
    # Flatten sequence of actions.
    a = action_[:, :-1].view(N, -1)
    n_a = action_[:, 1:].view(N, -1)
    # Concatenate feature and action.
    fa = torch.cat([f, a], dim=-1)
    n_fa = torch.cat([n_f, n_a], dim=-1)
    return fa, n_fa


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


def build_mlp(
    input_dim,
    output_dim,
    hidden_units=[64, 64],
    hidden_activation=nn.Tanh(),
    output_activation=None,
):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_gaussian_log_prob(log_std, noise):
    return (-0.5 * noise.pow(2) - log_std).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_std.size(-1)


def calculate_log_pi(log_std, noise, action):
    gaussian_log_prob = calculate_gaussian_log_prob(log_std, noise)
    return gaussian_log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(mean, log_std):
    noise = torch.randn_like(mean)
    action = torch.tanh(mean + noise * log_std.exp())
    return action, calculate_log_pi(log_std, noise, action)


def calculate_kl_divergence(p_mean, p_std, q_mean, q_std):
    #print(p_mean.shape, p_std.shape, q_mean.shape, q_std.shape)
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
import pickle
from collections import deque
import numpy as np
import random

class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, job_range, machine_range):
        self.step_count_list = dict()
        self.total_buffer=dict()
        self.problem_size_list = list()
        for j in range(job_range[0], job_range[1]):
            for i in range(machine_range[0], machine_range[1]):
                self.problem_size_list.append('{}_{}'.format(j, i))
        self.step_count = dict()
        for problem_size in self.problem_size_list:
            buffer = list()
            for _ in range(4):
                buffer.append(deque(maxlen=buffer_size))
            self.total_buffer[problem_size] = buffer
            self.step_count_list[problem_size] = []
            self.step_count[problem_size] = 0


        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.eligible_test = False


    def pop(self):
        self.buffer.pop()

    def memory(self, jobs_data, action_sequences, pi_olds, makespan, job_size, machine_size):
        problem_size = '{}_{}'.format(job_size, machine_size)
        buffer = self.total_buffer[problem_size]
        buffer[0].append(jobs_data)
        buffer[1].append(action_sequences)
        buffer[2].append(pi_olds)
        buffer[3].append(makespan)
        #print('전', self.buffer_size,self.step_count[problem_size] )
        if self.step_count[problem_size] < self.buffer_size - 1:
            self.step_count_list[problem_size].append(self.step_count[problem_size])
            self.step_count[problem_size] += 1
            #print('후', self.buffer_size, self.step_count[problem_size])


    def generating_mini_batch(self, datas, batch_idx, cat):
        for s in batch_idx:
            if cat == 'jobs_data':
                yield datas[0][s]
            if cat == 'action_sequences':
                yield datas[1][s]
            if cat == 'pi_olds':
                yield datas[2][s]
            if cat == 'makespan':
                yield datas[3][s]



    def sample(self):
        if self.eligible_test ==False:
            eligible_problem_sizes = \
            [
                ps for ps, count in self.step_count.items()
                if count > self.batch_size
            ]

            if len(eligible_problem_sizes) == len(self.problem_size_list):
                self.eligible_test = True
        else:
            eligible_problem_sizes = self.problem_size_list

        sampled_problem_size = random.sample(eligible_problem_sizes, k = 1)[0]

        step_count_list = self.step_count_list[sampled_problem_size][:]
        buffer = self.total_buffer[sampled_problem_size]
        step_count_list.pop()
        sampled_batch_idx = random.sample(step_count_list, self.batch_size)

        #node_feature, edge_index, action_sequences, pi_olds, makespan, job_size, machine_size
        jobs_data = self.generating_mini_batch(buffer, sampled_batch_idx, cat='jobs_data')
        jobs_data  = list(jobs_data )
        action_sequences = self.generating_mini_batch(buffer, sampled_batch_idx, cat='action_sequences')
        action_sequences = list(action_sequences)

        pi_olds = self.generating_mini_batch(buffer, sampled_batch_idx, cat='pi_olds')
        pi_olds= list(pi_olds)

        makespan = self.generating_mini_batch(buffer, sampled_batch_idx, cat='makespan')
        makespan = list(makespan)

        return jobs_data , action_sequences, pi_olds, makespan, sampled_problem_size

