import math

import torch
import torch.nn as nn

def initialize_weight(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def calculate_bottleneck_index(scheduler):
    """
    Bottleneck Index (Ib) 계산

    논문 공식:
    Ib_ik = (m_ik - 1) / (n - 1)
    Ib = (1/m) * Σ_k Σ_i Ib_ik

    여기서 m_ik는 k번째 operation position에서 machine i를 사용하는 job의 수

    Args:
        scheduler: AdaptiveScheduler 객체

    Returns:
        float: Bottleneck Index (0 ≤ Ib ≤ 1)
    """
    n = scheduler.num_job  # number of jobs
    m = scheduler.num_mc  # number of machines

    if n <= 1:
        return 0.0

    # 각 position k에서 각 machine i가 사용되는 횟수 계산
    total_ib = 0.0

    # 각 operation position k에 대해 (최대 operation 수는 machine 수와 같음)
    for k in range(m):
        for i in range(m):
            m_ik = 0  # k번째 position에서 machine i를 사용하는 job 수

            # 각 job에 대해 k번째 operation의 machine 확인
            for job_idx in range(n):
                if k < len(scheduler.ms[job_idx]):  # job에 k번째 operation이 존재하는 경우
                    # machine sequence는 1-based이므로 1을 빼서 0-based로 변환
                    machine_at_k = scheduler.ms[job_idx][k] - 1
                    if machine_at_k == i:
                        m_ik += 1

            # Ib_ik 계산
            if m_ik > 0:
                ib_ik = (m_ik - 1) / (n - 1)
                total_ib += ib_ik

    # 전체 Bottleneck Index 계산
    ib = total_ib / m
    return ib


def calculate_flow_shop_index(scheduler):
    """
    Flow Shop Index (If) 계산

    논문 공식:
    If_ik = (n_ik - 1) / (n - 1)
    If = (1/(m-1)) * Σ_i Σ_k If_ik

    여기서 n_ik는 machine i에서 처리된 후 바로 machine k로 이동하는 job의 수

    Args:
        scheduler: AdaptiveScheduler 객체

    Returns:
        float: Flow Shop Index (0 ≤ If ≤ 1)
    """
    n = scheduler.num_job  # number of jobs
    m = scheduler.num_mc  # number of machines

    if n <= 1 or m <= 1:
        return 0.0

    # machine i에서 machine k로 바로 이동하는 job 수를 저장하는 matrix
    transition_count = [[0 for _ in range(m)] for _ in range(m)]

    # 각 job의 machine sequence를 분석하여 transition 계산
    for job_idx in range(n):
        machine_sequence = scheduler.ms[job_idx]

        # 연속된 operation 간의 transition 계산
        for op_idx in range(len(machine_sequence) - 1):
            # machine sequence는 1-based이므로 1을 빼서 0-based로 변환
            current_machine = machine_sequence[op_idx] - 1
            next_machine = machine_sequence[op_idx + 1] - 1

            transition_count[current_machine][next_machine] += 1

    # Flow Shop Index 계산
    total_if = 0.0

    for i in range(m):
        for k in range(m):
            if i != k:  # 같은 machine으로의 transition은 제외
                n_ik = transition_count[i][k]
                if n_ik > 0:
                    if_ik = (n_ik - 1) / (n - 1)
                    total_if += if_ik

    # 전체 Flow Shop Index 계산
    if_index = total_if / (m - 1) if m > 1 else 0.0
    return if_index

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

def gumbel_softmax_hard(
    logits,
    tau = 1.0,
    dim: int = -1,
) :


    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(
        logits, memory_format=torch.legacy_contiguous_format
    ).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret, y_soft

