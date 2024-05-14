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



"""
FT10 Problem(Fisher and Thompson, 1963)
"""


class Job:
    def __init__(self, env, id, job_data):
        self.env = env
        self.id = id
        self.n = len(job_data)
        self.m = [job_data[i][0] for i in range(len(job_data))]
        self.d = [job_data[i][1] for i in range(len(job_data))]
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
            yield self.o[i].finished
        self.finished.succeed()


class Operation:
    def __init__(self, env, job_id, op_id, machine, duration):
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
        self.id = id
        self.env = env
        self.machine = simpy.Resource(self.env, capacity=1)
        self.queue = simpy.Store(self.env)
        self.available_num = 0
        self.waiting_operations = {}
        self.availability = self.env.event()
        self.availability.succeed()
        self.workingtime_log = []
        self.execute()

    def execute(self):
        self.env.process(self.processing())

    def processing(self):
        while True:
            op = yield self.queue.get()
            self.available_num += 1
            yield self.availability
            self.availability = self.env.event()
            yield op.waiting  # waiting이 succeed로 바뀔 떄까지 기다림
            starting_time = self.env.now
            op.starting_time = starting_time
            yield self.env.timeout(op.duration)
            finishing_time = self.env.now
            op.finishing_time = finishing_time
            op.finished.succeed()
            self.workingtime_log.append((op.job_id, starting_time, finishing_time))
            self.availability.succeed()


class AdaptiveScheduler:
    def __init__(self, input_data):
        self.jobs_data = input_data
        self.input_data = input_data
        self.num_mc = len(self.input_data)   # number of machines
        self.num_job = len(self.input_data)  # number of jobs
        self.pt = [[ops[1] for ops in job] for job in self.input_data] # processing_time
        self.ms = [[ops[0]+1 for ops in job] for job in self.input_data] # job 별 machine sequence
        self.j_keys = [j for j in range(self.num_job)]
        self.key_count = {key: 0 for key in self.j_keys}
        self.j_count =   {key: 0 for key in self.j_keys}
        self.m_keys =  [j + 1 for j in range(self.num_mc)]
        self.m_count = {key: 0 for key in self.m_keys}

    def adaptive_run(self, est_holder, fin_holder, i= None):

        estI_list=list()
        gentI_list=list()
        for I in range(self.num_job):
            try:
                gen_tI = int(self.pt[I][self.key_count[I]])
                gen_mI = int(self.ms[I][self.key_count[I]])
                estI = max(self.j_count[I], self.m_count[gen_mI])
                estI_list.append(estI)
                gentI_list.append(estI+gen_tI)
            except IndexError:
                pass

        for I in range(self.num_job):
            try:
                gen_tI = int(self.pt[I][self.key_count[I]])
                gen_mI = int(self.ms[I][self.key_count[I]])
                estI = max(self.j_count[I], self.m_count[gen_mI])

                index_of_one = (est_holder[I] == 1)
                if len(estI_list)>0 and np.max(estI_list)!=0:
                    est_holder[I][index_of_one] = estI/np.max(estI_list)
                    fin_holder[I][index_of_one] = (estI+gen_tI)/np.max(gentI_list)
                else:pass
                estI_list.append(estI)
                gentI_list.append(estI+gen_tI)
            except IndexError:
                pass

        if i != None:
            gen_t = int(self.pt[i][self.key_count[i]])
            gen_m = int(self.ms[i][self.key_count[i]])
            self.j_count[i] = self.j_count[i] + gen_t
            self.m_count[gen_m] = self.m_count[gen_m] + gen_t
            if self.m_count[gen_m] < self.j_count[i]:
                self.m_count[gen_m] = self.j_count[i]
            elif self.m_count[gen_m] > self.j_count[i]:
                self.j_count[i] = self.m_count[gen_m]
            self.key_count[i] = self.key_count[i] + 1


        return est_holder, est_holder

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
        self.num_mc = len(self.input_data)   # number of machines
        self.num_job = len(self.input_data)  # number of jobs
        self.pt = [[ops[1] for ops in job] for job in self.input_data] # processing_time
        self.ms = [[ops[0]+1 for ops in job] for job in self.input_data] # job 별 machine sequence
        self.j_keys = [j for j in range(self.num_job)]
        self.key_count = {key: 0 for key in self.j_keys}
        self.j_count =   {key: 0 for key in self.j_keys}
        self.m_keys =  [j + 1 for j in range(self.num_mc)]
        self.m_count = {key: 0 for key in self.m_keys}

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

    def get_fully_connected_edge_index(self):
        #jk = 0
        n = len(self.jobs_data)*len(self.jobs_data)
        rows = [i // n for i in range(n ** 2)]
        cols = [i % n for i in range(n ** 2)]
    #return [rows, cols]

        return [rows, cols]


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