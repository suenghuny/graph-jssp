import copy
import collections
import networkx as nx
import numpy as np
import simpy
import random
import time
from cfg import get_cfg
cfg = get_cfg()
start = time.time()
random.seed(525312344)

class Job:
    def __init__(self, env, id, job_data):
        self.env = env
        self.id = id
        self.n = len(job_data)
        self.m = [job_data[i][0] for i in range(len(job_data))]  # machine
        self.d = [job_data[i][1] for i in range(len(job_data))]  # duration
        self.o = [Operation(env, self.id, i, self.m[i], self.d[i]) for i in range(self.n)]
        self.completed = 0
        self.scheduled = 0
        self.finished = env.event()
        self.execute()

    def execute(self):
        self.env.process(
            self.next_operation_ready()
        )

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
            self.workingtime_log.append((op.job_id, op.starting_time, finishing_time, op.op_id))
            self.availability.succeed()


class Scheduler:
    def __init__(self, jobs_data):
        self.env = simpy.Environment()
        self.job_list = []
        self.jobs_data = jobs_data
        self.machine_list = []
        self.c_max = 0
        for i in range(len(self.jobs_data)):
            self.job_list.append(Job(self.env, i, jobs_data[i]))
        for i in range(len(self.jobs_data)):
            self.machine_list.append(Machine(self.env, i))

    def partial_run(self, partial_sequence):
        self.schedule(partial_sequence)
        self.env.process(self.evaluate())
        self.env.run()
        return self.c_max


    def run(self, sequence):
        self.schedule(sequence)
        self.env.process(self.evaluate())
        self.env.run()
        return self.c_max

    def get_fully_connected_edge_index(self):
        n = len(self.jobs_data)*len(self.jobs_data)
        rows = [i // n for i in range(n ** 2)]
        cols = [i % n for i in range(n ** 2)]
        return [rows, cols]

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

    def get_machine_sharing_edge_index(self):
        jk = 0
        machine_sharing = [[] for _ in range(len(self.job_list))]
        for job in self.jobs_data:
            for k in range(len(job)):
                ops = job[k]
                machine_sharing[ops[0]].append(jk)
                jk += 1
        edge_index = [[],[]]
        for machines in machine_sharing:
            for m in machines:
                for m_prime in machines:
                    if cfg.gnn_type == 'gcrl':
                        edge_index[0].append(jk)
                        edge_index[1].append(jk)
                    else:
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
                    if cfg.gnn_type == 'gcrl':
                        edge_index[0].append(jk)
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
                    if cfg.gnn_type == 'gcrl':
                        edge_index[0].append(jk)
                        edge_index[1].append(jk)
                    jk += 1
        return edge_index

    def schedule(self,sequence):
        for i in sequence:
            o_ = self.job_list[i].scheduled
            m_ = self.job_list[i].m[o_]
            self.machine_list[m_].queue.put(self.job_list[i].o[o_])
            self.job_list[i].scheduled += 1

    def evaluate(self):
        finished_jobs = [self.job_list[i].finished for i in range(len(self.job_list))]
        yield simpy.AllOf(self.env, finished_jobs)
        self.c_max = self.env.now
