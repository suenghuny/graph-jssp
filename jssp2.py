import copy
import collections
import numpy as np
import simpy
import random
import pandas as pd
from collections import defaultdict
import os
import pickle
import time

pt_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Processing Time", index_col=[0], engine = 'openpyxl')
ms_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col=[0], engine = 'openpyxl')


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


        self.mask1 =  [[0 for i in range(self.num_mc)] for j in range(self.num_job)]
        self.mask2 =  [[1 for i in range(self.num_mc)] for j in range(self.num_job)]




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

        self.mask1 = [[0 for i in range(self.num_mc)] for j in range(self.num_job)]
        self.mask2 = [[1 for i in range(self.num_mc)] for j in range(self.num_job)]

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
                sum_ops_o.append(0)
                sum_ops_o = sum(sum_ops_o)
                node_features.append([float(ops[1])/np.max(empty), sum_ops_o/sum_ops, (o+1)/len(job)])
        node_features.append([0., 1., 1])
        node_features.append([0., 0., 0])
        return node_features

    def get_fully_connected_edge_index(self):
        n = len(self.jobs_data)*len(self.jobs_data)
        rows = [i // n for i in range(n ** 2)]
        cols = [i % n for i in range(n ** 2)]
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