import copy
import collections
import numpy as np
import simpy
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
from collections import defaultdict
import os
import pickle
import time

import networkx as nx
pt_tmp = pd.read_excel("JSP_dataset.xlsx", sheet_name="Processing Time", index_col=[0], engine = 'openpyxl')
ms_tmp = pd.read_excel(
         "JSP_dataset.xlsx", sheet_name="Machines Sequence", index_col=[0], engine = 'openpyxl')


class GraphVisualizer:
    def __init__(self, instance_data):
        self.processing_time, self.machine_sequence = instance_data
        self.G = nx.DiGraph()
        self.pos = {}
        self.edge_labels = None
        self._create_disjunctive_graph()

    def _create_disjunctive_graph(self):
        dummy_start = 'Start'
        dummy_end = 'End'
        self.G.add_node(dummy_start)
        self.G.add_node(dummy_end)
        self.flattend_processing_time_dict = dict()
        k = 0
        num_machine = np.max(np.array(self.machine_sequence))
        self.flattend_machine_sequence_list = [[] for _ in range(num_machine)]
        self.flattend_machine_allocation_dict = dict()
        for j, machine_sequence_list in enumerate(self.machine_sequence):
            for i in range(len(machine_sequence_list)):
                m = machine_sequence_list[i]
                self.flattend_machine_sequence_list[m-1].append(k)
                self.flattend_machine_allocation_dict[k] = m-1
                k += 1
        k = 0
        for j, processing_time_list in enumerate(self.processing_time):
            for i in range(len(processing_time_list)):
                self.G.add_node(k)
                self.pos[k] = (i+1, j)  # 노드 위치 설정 (x: 노드 인덱스, y: 고정)
                self.flattend_processing_time_dict[k] = processing_time_list[i]
                if i == 0:
                    self.G.add_edge(dummy_start, k, weight=0)
                if i < len(processing_time_list) - 1:
                    self.G.add_edge(k, k+1,
                                    weight=-1* processing_time_list[i])
                if i == len(processing_time_list) - 1:
                    self.G.add_edge(k, dummy_end, weight=-1* processing_time_list[i])
                k+=1
        self.pos[dummy_start] = (0, len(processing_time_list)/2)
        self.pos[dummy_end] = (len(self.processing_time)+2, len(processing_time_list)/2)
        self.edge_labels = {(u, v): d['weight'] for u, v, d in self.G.edges(data=True)}


    def get_earliest_start_and_finish_time(self, available_operations):
        avail_nodes = np.array(available_operations)
        avail_nodes_indices = np.where(avail_nodes == 1)[0]
        earliest_start_time = np.zeros(len(self.processing_time)*len(self.machine_sequence)) # 여기는 검토해봐야함
        earliest_finish_time = np.zeros(len(self.processing_time)*len(self.machine_sequence)) # 여기는 검토해봐야함
        est_list = list()
        fin_list = list()
        for operation in avail_nodes_indices:
            est = -1*self.get_longest_path(fm = 'Start', to = operation)
            efin = est + self.flattend_processing_time_dict[operation]
            earliest_start_time[operation] = est
            earliest_finish_time[operation] = efin
            est_list.append(est)
            fin_list.append(efin)
        if np.max(est_list) == 0:
            earliest_start_time = np.zeros(len(self.processing_time)*len(self.machine_sequence)) # 여기는 검토해봐야함
        else:
            earliest_start_time = earliest_start_time / np.max(est_list)
        earliest_finish_time = earliest_finish_time / np.max(fin_list)

        return earliest_start_time, earliest_finish_time




    def get_longest_path(self, fm, to):
        shortest_path_length = nx.bellman_ford_path_length(self.G, source=fm, target=to, weight='weight')
        return shortest_path_length




    def add_selected_operation(self, k):
        m = self.flattend_machine_allocation_dict[k]
        machine_sharing_operation = self.flattend_machine_sequence_list[m]
        processing_time = self.flattend_processing_time_dict[k]
        for k_prime in machine_sharing_operation:
            if k != k_prime:
                self.G.add_edge(k, k_prime, weight=-1*processing_time)
        if k in self.flattend_machine_sequence_list[m]:
            self.flattend_machine_sequence_list[m].remove(k)

    def get_lower_bound(self, k):
        m = self.flattend_machine_allocation_dict[k]
        machine_sharing_operation = self.flattend_machine_sequence_list[m]
        processing_time = self.flattend_processing_time_dict[k]
        for k_prime in machine_sharing_operation:
            if k != k_prime:
                self.G.add_edge(k, k_prime, weight=-1*processing_time)


        longest_path = -1*self.get_longest_path('Start', 'End')
        for k_prime in machine_sharing_operation:
            if k != k_prime:
                self.G.remove_edge(k, k_prime)
        return longest_path


    def show(self):
        fig, ax = plt.subplots(figsize=(25, 15))  # 명시적으로 Figure와 Axes 객체 생성
        nx.draw(self.G, self.pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10,
                font_weight='bold', arrowsize=20, ax=ax)  # Axes 객체를 명시적으로 전달
        nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=self.edge_labels, ax=ax)  # Axes 객체를 명시적으로 전달
        plt.title('Graph Representation of the Given Lists')
        plt.show()

class AdaptiveScheduler:
    def __init__(self, input_data):
        self.jobs_data = input_data
        self.input_data = input_data
        self.num_mc = len(self.input_data[0])   # number of machines
        self.num_job = len(self.input_data)     # number of jobs
        self.pt = [[ops[1] for ops in job] for job in self.input_data] # processing_time
        self.ms = [[ops[0]+1 for ops in job] for job in self.input_data] # job 별 machine sequence
        self.j_keys = [j for j in range(self.num_job)]
        self.key_count = {key: 0 for key in self.j_keys}
        self.j_count = {key: 0 for key in self.j_keys}
        self.m_keys =  [j + 1 for j in range(self.num_mc)]
        self.m_count = {key: 0 for key in self.m_keys}
        self.num_ops = self.num_job*self.num_mc

        self.job_id_ops = list()
        j = 0
        for job in self.input_data:
            for ops in job:
                self.job_id_ops.append(j)
            j+=1

        self.mask1 =  [[0 for _ in range(self.num_mc)] for _ in range(self.num_job)]
        self.mask2 =  [[1 for _ in range(self.num_mc)] for _ in range(self.num_job)]
        data = self.pt
        self.graph = GraphVisualizer((data, self.ms))

    def add_selected_operation(self, k):
        self.graph.add_selected_operation(k)

    def get_longest_path(self):
        longest_path_length = self.graph.get_longest_path(fm = 'Start', to='End')
        return longest_path_length

    def show(self):
        self.graph.show()

    def get_earliest_start_and_finish_time(self, available_operations):
        earliest_start_time, earliest_finish_time = self.graph.get_earliest_start_and_finish_time(available_operations)
        return earliest_start_time, earliest_finish_time

    def get_lower_bound(self, avail_nodes_indices):
        lower_bound_list = list()
        #print(avail_nodes_indices)
        for k_prime in avail_nodes_indices:
            lower_bound = self.graph.get_lower_bound(k_prime)
            lower_bound_list.append(lower_bound)
        return lower_bound_list

    def adaptive_run(self, est_holder, fin_holder, i= None):
        if i != None:                                         # 이전 스텝에서 선택된 Job(Operation)에 대한 update
            try:
                gen_t = int(self.pt[i][self.key_count[i]])        # 선택된 operation에 대한 processing time 선택
                gen_m = int(self.ms[i][self.key_count[i]])        # 선택된 operation에 대한 machine_sequence 선택
                self.j_count[i] = self.j_count[i] + gen_t          # Job i에 대한 누적 작업 완료시간 업데이트
                self.m_count[gen_m] = self.m_count[gen_m] + gen_t  # Machine gen_m에 대한 누적 작업 완료시간 업데이트
                if self.m_count[gen_m] < self.j_count[i]:
                    self.m_count[gen_m] = self.j_count[i]
                elif self.m_count[gen_m] > self.j_count[i]:
                    self.j_count[i] = self.m_count[gen_m]          # if 및 elif 문은 각각의 누적 작업 완료시간을 큰 녀석으로 업데이트 한다는 의미
                self.key_count[i] = self.key_count[i] + 1          # 해당 Job이 몇번 선택되었는지 count하는 것 업데이트
            except IndexError:
                pass

        estI_list = list()
        gentI_list = list()
        for I in range(self.num_job): # 아직 선택되지 않은 녀석들(선택될 가능성이 있는 애들)에 대한 이야기
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



    def rollout_run(self, t_th, avail_nodes_indices):
        makespan_list = list()
        for k_prime in avail_nodes_indices:
            i_prime = self.job_id_ops[k_prime]
            key_count = deepcopy(self.key_count)
            j_count = deepcopy(self.j_count)
            m_count = deepcopy(self.m_count)
            if i_prime != None:                                         # 이전 스텝에서 선택된 Job(Operation)에 대한 update
                try:
                    gen_t = int(self.pt[i_prime][key_count[i_prime]])        # 선택된 operation에 대한 processing time 선택
                    gen_m = int(self.ms[i_prime][key_count[i_prime]])        # 선택된 operation에 대한 machine_sequence 선택
                    j_count[i_prime] = j_count[i_prime] + gen_t          # Job i에 대한 누적 작업 완료시간 업데이트
                    m_count[gen_m] = m_count[gen_m] + gen_t  # Machine gen_m에 대한 누적 작업 완료시간 업데이트
                    if m_count[gen_m] < j_count[i_prime]:
                        m_count[gen_m] = j_count[i_prime]
                    elif m_count[gen_m] > j_count[i_prime]:
                        j_count[i_prime] = m_count[gen_m]          # if 및 elif 문은 각각의 누적 작업 완료시간을 큰 녀석으로 업데이트 한다는 의미
                    key_count[i_prime] = key_count[i_prime] + 1
                except IndexError:pass

            for _ in range(self.num_ops-t_th-1):
                pt_list = list()
                for i in range(self.num_job):
                    try:
                        pt = self.pt[i][key_count[i]]
                        pt_list.append(pt)
                    except IndexError as IE:
                        pt_list.append(float('inf'))

                if len(pt_list)>0:
                    i = np.argmin(pt_list)
                    gen_t = int(self.pt[i][key_count[i]])
                    gen_m = int(self.ms[i][key_count[i]])
                    j_count[i] = j_count[i] + gen_t
                    m_count[gen_m] = m_count[gen_m] + gen_t
                    if m_count[gen_m] < j_count[i]:
                        m_count[gen_m] = j_count[i]
                    elif m_count[gen_m] > j_count[i]:
                        j_count[i] = m_count[gen_m]
                    key_count[i] = key_count[i] + 1


            makespan = max(j_count.values())
            makespan_list.append(makespan)
        return makespan_list


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
        self.num_mc = len(self.input_data[0])   # number of machines
        self.num_job = len(self.input_data)     # number of jobs
        self.pt = [[ops[1] for ops in job] for job in self.input_data] # processing_time
        self.ms = [[ops[0]+1 for ops in job] for job in self.input_data] # job 별 machine sequence
        self.j_keys = [j for j in range(self.num_job)]
        self.key_count = {key: 0 for key in self.j_keys}
        self.j_count = {key: 0 for key in self.j_keys}
        self.m_keys =  [j + 1 for j in range(self.num_mc)]
        self.m_count = {key: 0 for key in self.m_keys}
        self.num_ops = self.num_job*self.num_mc
        self.job_id_ops = list()
        j = 0
        for job in self.input_data:
            for ops in job:
                self.job_id_ops.append(j)
            j+=1
        self.mask1 =  [[0 for _ in range(self.num_mc)] for _ in range(self.num_job)]
        self.mask2 =  [[1 for _ in range(self.num_mc)] for _ in range(self.num_job)]
        data = self.pt
        self.graph = GraphVisualizer((data, self.ms))


    def get_node_feature(self):
        node_features = []
        empty = list()
        for j in range(len(self.jobs_data)):
            job = self.jobs_data[j]
            for o in range(len(job)):
                ops = job[o]
                empty.append(ops[1])

        jk=0
        empty2 = list()
        for j in range(len(self.jobs_data)):
            job = self.jobs_data[j]
            sum_ops = sum([float(job[o][1]) for o in range(len(job))])
            empty2.append(sum_ops)


        for j in range(len(self.jobs_data)):
            job = self.jobs_data[j]
            sum_ops = sum([float(job[o][1]) for o in range(len(job))])

            for o in range(len(job)):
                ops = job[o]
                sum_ops_o = [float(job[k][1]) for k in range(0, o+1)]
                sum_ops_o.append(0)
                sum_ops_o = sum(sum_ops_o)
                node_features.append([
                                      float(ops[1]) / sum_ops,
                                      float(ops[1]) / np.max(empty),
                                      sum_ops_o/sum_ops,
                                      sum_ops / np.max(empty2),
                                      (o+1)/len(job)
                                     ])

        node_features.append([0., 1., 1, 0, 0])
        node_features.append([0., 0., 0, 0, 0])
        return node_features

    def get_fully_connected_edge_index(self):
        n = len(self.jobs_data)*len(self.jobs_data)
        rows = [i // n for i in range(n ** 2)]
        cols = [i % n for i in range(n ** 2)]
        return [rows, cols]


    def get_machine_sharing_edge_index(self):
        jk = 0
        machine_sharing = [[] for _ in range(self.num_mc)]

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
                    jk += 1
                else:
                    edge_index[0].append(jk)
                    edge_index[1].append(jk-1)
                    edge_index[0].append(jk-1)
                    edge_index[1].append(jk)
                    jk += 1
        return edge_index