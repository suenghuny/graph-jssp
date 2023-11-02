import copy
import collections
import numpy as np
import simpy
import random
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import pandas as pd
from collections import defaultdict
import os
import pickle
import time

pt_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Processing Time", index_col=[0])
ms_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col=[0])
print()



start = time.time()


random.seed(42)

"""
FT10 Problem(Fisher and Thompson, 1963)
"""
jobs_data = [
    [(0, 29), (1, 78), (2, 9), (3, 36), (4, 49), (5, 11), (6, 62), (7, 56), (8, 44), (9, 21)],
    [(0, 43), (2, 90), (4, 75), (9, 11), (3, 69), (1, 28), (6, 46), (5, 46), (7, 72), (8, 30)],
    [(1, 91), (0, 85), (3, 39), (2, 74), (8, 90), (5, 10), (7, 12), (6, 89), (9, 45), (4, 33)],
    [(1, 81), (2, 95), (0, 71), (4, 99), (6, 9), (8, 52), (7, 85), (3, 98), (9, 22), (5, 43)],
    [(2, 14), (0, 6), (1, 22), (5, 61), (3, 26), (4, 69), (8, 21), (7, 49), (9, 72), (6, 53)],
    [(2, 84), (1, 2), (5, 52), (3, 95), (8, 48), (9, 72), (0, 47), (6, 65), (4, 6), (7, 25)],
    [(1, 46), (0, 37), (3, 61), (2, 13), (6, 32), (5, 21), (9, 32), (8, 89), (7, 30), (4, 55)],
    [(2, 31), (0, 86), (1, 46), (5, 74), (4, 32), (6, 88), (8, 19), (9, 48), (7, 36), (3, 79)],
    [(0, 76), (1, 69), (3, 76), (5, 51), (2, 85), (9, 11), (6, 40), (7, 89), (4, 26), (8, 74)],
    [(1, 85), (0, 13), (2, 61), (6, 7), (8, 64), (9, 76), (5, 47), (3, 52), (4, 90), (7, 45)]
] # machine, procesing time

NUM_POPULATION = 500
P_CROSSOVER = 0.5
P_MUTATION = 0.9
N_JOBS = 10
N_OP_per_JOBS = 10
N_OPERATION = 100
N_MACHINE = 10


TERMINATION = 200

# for i in range(len(jobs_data)):
#     for j in range(len(jobs_data[i])):
#         # print(jobs_data[i][j][0])
#         max_machine = jobs_data[i][j][0]
#         if max_machine > N_MACHINE:
#             N_MACHINE = max_machine
# N_MACHINE += 1
# N_JOBS = len(jobs_data)  # 10
# for i in range(len(jobs_data)):
#     N_OPERATION += len(jobs_data[i])

class Individual:
    def __init__(self, seq):
        self.seq_op = seq  # 중복을 허용하지 않는 순열
        self.seq_job = self.repeatable_permuation()  # 중복을 허용하는 순열
        self.makespan = self.get_makespan()

    def repeatable_permuation(self):
        cumul = 0
        sequence_ = np.array(self.seq_op)
        for i in range(N_JOBS):
            for j in range(N_OP_per_JOBS):
                sequence_ = np.where((sequence_ >= cumul) &
                                     (sequence_ < cumul + N_OP_per_JOBS), i, sequence_)
            cumul += N_OP_per_JOBS
        sequence_ = sequence_.tolist()
        return sequence_

    def get_makespan(self):

        scheduler_ = Scheduler(jobs_data, N_MACHINE, self.seq_job)  # sequence = list
        makespan = scheduler_.c_max
        del scheduler_
        return makespan

def non_repetitive_permutation(p_1):
    # 중복을 허용하지 않는 순열로 재구성
    p1_idx_list = [list(filter(lambda e: p_1[e] == i, range(len(p_1)))) for i in range(N_OP_per_JOBS)]  # 10 is the number of operations

    for k in range(N_JOBS):  # if k = 1,
        indices = p1_idx_list[k]  # indices where p_1 is 8
        t = 0
        for idx in indices:
            p_1[idx] = k * N_OP_per_JOBS + t  # the permutation [8, 8, 8, ... ] is converted to [80, 81, 82, ... ]
            t += 1

    return p_1

class Job:
    def __init__(self, env, id, job_data):
        self.env = env
        self.id = id
        # print('Job ',id,' generated!')
        self.n = len(job_data)
        self.m = [job_data[i][0] for i in range(len(job_data))]  # machine
        # print('Job %d Machine list : ' % self.id, self.m)
        self.d = [job_data[i][1] for i in range(len(job_data))]  # duration
        self.o = [Operation(env, self.id, i, self.m[i], self.d[i]) for i in range(self.n)]
        self.completed = 0
        self.scheduled = 0
        # List to track waiting operations
        self.finished = env.event()

        self.execute()

    def execute(self):
        self.env.process(self.next_operation_ready())

    def next_operation_ready(self):
        for i in range(self.n):
            self.o[i].waiting.succeed()
            # print('%d Operation %d%d is completed, waiting for Operation %d%d to be completed ...' % (env.now, self.id, self.completed, self.id, self.completed+1))
            yield self.o[i].finished
            # print('%d Operation %d%d Finished!' % (self.env.now, self.id, self.completed))
        # print('%d Job %d All Operations Finished!' % (self.env.now, self.id))
        self.finished.succeed()


class Operation:
    def __init__(self, env, job_id, op_id, machine, duration):
        # print('Operation %d%d generated!' % (job_id, op_id))
        self.env = env
        self.job_id = job_id
        self.op_id = op_id
        self.machine = machine
        self.duration = duration
        self.starting_time = 0.0
        self.finishing_time = 0.0
        self.waiting = self.env.event()
        self.finished = self.env.event()


class Machine:
    def __init__(self, env, id):
        # print('Machine ',id,' generated!')
        self.id = id
        self.env = env
        self.machine = simpy.Resource(self.env, capacity=1)
        self.queue = simpy.Store(self.env)
        self.available_num = 0
        self.waiting_operations = {}
        # self.availability = [self.env.event() for i in range(100)]
        # self.availability[0].succeed()
        self.availability = self.env.event()
        self.availability.succeed()
        self.workingtime_log = []

        self.execute()

    def execute(self):
        self.env.process(self.processing())

    def processing(self):
        while True:
            op = yield self.queue.get()
            # print('%d : Job %d is waiting on M%d' % (self.env.now, job.id, self.id))

            # yield self.availability[self.available_num]
            self.available_num += 1
            yield self.availability
            self.availability = self.env.event()
            # print('M%d Usage Count : %d' %(self.id, self.available_num)
            yield op.waiting  # waiting이 succeed로 바뀔 떄까지 기다림
            starting_time = self.env.now
            op.starting_time = starting_time
            yield self.env.timeout(op.duration)
            finishing_time = self.env.now
            op.finishing_time = finishing_time
            op.finished.succeed()
            self.workingtime_log.append((op.job_id, starting_time, finishing_time))
            self.availability.succeed()


class Scheduler:
    def __init__(self, input_data):
        self.jobs_data = input_data
        self.num_mc = len(input_data)   # number of machines
        self.num_job = len(input_data)  # number of jobs


        self.pt = [[ops[1] for ops in job] for job in input_data] # processing_time
        self.ms = [[ops[0] for ops in job] for job in input_data] # job 별 machine sequence

        self.j_keys = [j for j in range(self.num_job)]
        self.key_count = {key: 0 for key in self.j_keys}
        self.j_count =   {key: 0 for key in self.j_keys}
        self.m_keys =  [j + 1 for j in range(self.num_mc)]
        self.m_count = {key: 0 for key in self.m_keys}

    def run(self, sequence):
        for i in sequence:
            gen_t = int(self.pt[i][self.key_count[i]])
            gen_m = int(self.ms[i][self.key_count[i]])
            self.j_count[i] = self.j_count[i] + gen_t
            self.m_count[gen_m] = self.m_count[gen_m] + gen_t
            if self.m_count[gen_m] < self.j_count[i]:
                self.m_count[gen_m] = self.j_count[i]
            elif self.m_count[gen_m] > self.j_count[i]:
                self.j_count[i] = self.m_count[gen_m]
            self.key_count[i] = self.key_count[i] + 1
        makespan = max(self.j_count.values())
        return makespan

    def reset(self):
        result = []
        for idx, row in ms_tmp.iterrows():
            empty = list()
            for col in ms_tmp.columns:
                o = row[col] - 1
                t = pt_tmp.at[idx, col]
                empty.append((o, t))
            result.append(empty)
        return result

    def get_node_feature(self):
        node_features = []
        empty = list()
        for j in range(len(self.jobs_data)):
            job = self.jobs_data[j]
            for o in range(len(job)):
                ops = job[o]
                empty.append(ops[1])


        for j in range(len(self.jobs_data)):
            job = self.jobs_data[j]

            sum_ops = sum([float(job[o][1]) for o in range(len(job))])
            for o in range(len(job)):
                ops = job[o]
                sum_ops_o = [float(job[k][1]) for k in range(0, o+1)]
                #print([float(job[k][1]) for k in range(0, o+1)])
                sum_ops_o.append(0)
                sum_ops_o = sum(sum_ops_o)
                node_features.append([float(ops[1])/np.max(empty), sum_ops_o/sum_ops, (o+1)/len(job)])
                #print([float(ops[1])/np.max(empty), sum_ops_o/sum_ops, (o+1)/len(job)])
                #node_features.append([float(ops[1]) / np.max(empty), sum_ops_o / sum_ops])

                #print([float(ops[1])/np.max(empty), sum_ops_o/sum_ops, (o+1)/len(job)])
        node_features.append([0., 1., 1])
        node_features.append([0., 0., 0])
        return node_features


    def get_machine_sharing_edge_index(self):
        jk = 0
        machine_sharing = [[] for _ in range(len(self.jobs_data))]
        for job in self.jobs_data:
            for k in range(len(job)):
                ops = job[k]
                machine_sharing[ops[0]].append(jk)
                jk += 1
        edge_index = [[],[]]
        for machines in machine_sharing:
            for m in machines:
                for m_prime in machines:
                    if m != m_prime:
                        edge_index[0].append(m)
                        edge_index[1].append(m_prime)


        return edge_index
        # print(machine_sharing)
        # print(edge_index)


    def get_edge_index_precedence(self):
        jk = 0
        edge_index = [[],[]]
        for job in self.jobs_data:
            for k in range(len(job)):
                if k == len(job)-1:
                    edge_index[0].append(len(self.jobs_data)*len(self.jobs_data[0]))
                    edge_index[1].append(jk)
                    edge_index[0].append(len(self.jobs_data)*len(self.jobs_data[0]))
                    edge_index[1].append(jk)
                    jk += 1
                else:
                    edge_index[0].append(jk)
                    edge_index[1].append(jk+1)
                    edge_index[0].append(jk+1)
                    edge_index[1].append(jk)
                    jk += 1
        return edge_index

    def get_edge_index_antiprecedence(self):
        jk = 0
        edge_index = [[],[]]
        for job in self.jobs_data:
            for k in range(len(job)):
                if k == 0:
                    edge_index[0].append(len(self.jobs_data)*len(self.jobs_data[0])+1)
                    edge_index[1].append(jk)
                    edge_index[0].append(jk)
                    edge_index[1].append(len(self.jobs_data)*len(self.jobs_data[0])+1)
                    jk += 1
                else:
                    edge_index[0].append(jk)
                    edge_index[1].append(jk-1)
                    edge_index[0].append(jk-1)
                    edge_index[1].append(jk)
                    jk += 1
        #print(edge_index)
        return edge_index