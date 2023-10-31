import copy
import collections
import numpy as np
import simpy
import random
import time

start = time.time()
random.seed(525312344)

"""
FT10 Problem(Fisher and Thompson, 1963)
"""
NUM_POPULATION = 500
P_CROSSOVER = 0.5
P_MUTATION = 0.9
N_JOBS = 10
N_OP_per_JOBS = 10
N_OPERATION = 100
N_MACHINE = 10
import itertools

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

            # print('M%d Usage Count : %d' %(self.id, self.available_num))

            yield op.waiting  # waiting이 succeed로 바뀔 떄까지 기다림
            starting_time = self.env.now
            op.starting_time = starting_time

            yield self.env.timeout(op.duration)
            finishing_time = self.env.now
            op.finishing_time = finishing_time
            op.finished.succeed()

            self.workingtime_log.append((op.job_id, op.starting_time, finishing_time, op.op_id))

            # print('%d Operation %d%d Finished on M%d!' % (self.env.now, op.job_id, op.op_id, self.id))
            # self.availability[self.available_num].succeed()
            self.availability.succeed()


class Scheduler:
    def __init__(self, jobs_data):
        self.env = simpy.Environment()
        self.job_list = []
        self.jobs_data = jobs_data
        self.machine_list = []
        self.c_max = 0
        for i in range(len(jobs_data)):
            self.job_list.append(Job(self.env, i, jobs_data[i]))
        for i in range(len(self.jobs_data)):
            self.machine_list.append(Machine(self.env, i))



    def run(self, sequence):
        self.schedule(sequence)
        self.env.process(self.evaluate())
        self.env.run()

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
                #print([float(ops[1])/np.max(empty), sum_ops_o/sum_ops, (o+1)/len(job)])
                node_features.append([float(ops[1]) / np.max(empty), sum_ops_o / sum_ops])
#                node_features.append([float(ops[1])/np.max(empty), sum_ops_o/sum_ops, (o+1)/len(job)])
                #print([float(ops[1])/np.max(empty), sum_ops_o/sum_ops, (o+1)/len(job)])
        node_features.append([0., 1.])
        node_features.append([0., 0.])
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
                #ops = job[k]
                if k == len(job)-1:
                    edge_index[0].append(len(self.jobs_data)*len(self.jobs_data[0]))
                    edge_index[1].append(jk)
                    # edge_index[0].append(len(self.jobs_data)*len(self.jobs_data[0]))
                    # edge_index[1].append(jk)
                    jk += 1
                else:
                    edge_index[0].append(jk)
                    edge_index[1].append(jk+1)
                    # edge_index[0].append(jk+1)
                    # edge_index[1].append(jk)
                    jk += 1
        return edge_index

    def get_edge_index_antiprecedence(self):
        jk = 0
        edge_index = [[],[]]
        for job in self.jobs_data:
            for k in range(len(job)):
                if k == 0:
                    # edge_index[0].append(len(self.jobs_data)*len(self.jobs_data[0])+1)
                    # edge_index[1].append(jk)
                    edge_index[0].append(jk)
                    edge_index[1].append(len(self.jobs_data)*len(self.jobs_data[0])+1)

                    jk += 1

                else:
                    #ops = job[k]
                    edge_index[0].append(jk)
                    edge_index[1].append(jk-1)
                    # edge_index[0].append(jk-1)
                    # edge_index[1].append(jk)
                    jk += 1
        #print(edge_index)
        return edge_index

    def schedule(self,sequence):
        for i in sequence:              # 0 0 1 2 1 2 0 1
            o_ = self.job_list[i].scheduled  # 0 1 0 0 1 1 2 2
            #print(len(self.job_list[i].m), o_)
            m_ = self.job_list[i].m[o_]      # 0 1 0 1 2 2 2 1
            self.machine_list[m_].queue.put(self.job_list[i].o[o_])
            self.job_list[i].scheduled += 1



    def evaluate(self):
        finished_jobs = [self.job_list[i].finished for i in range(len(self.job_list))]
        yield simpy.AllOf(self.env, finished_jobs)
        self.c_max = self.env.now
        # print("Total Makespan : ", self.c_max)


# Global Search
def global_search():
    # Generate Sequence
    result = []
    for i in range(2000):
        sequence = [i for i in range(N_OPERATION)]
        sequence = np.array(random.sample(sequence, len(sequence)))

        # Generating Job sequence from the random numbers (i.e. 0~4 refers to Job0, 5~9 refers to Job1, and so on.)
        ind = Individual(sequence.tolist())

        result.append([ind.seq_job, ind.makespan])

    makespan = [result[i][1] for i in range(len(result))]
    optimal = []
    for i in range(len(result)):

        if result[i][1] == min(makespan):
            print(result[i][0], 'makespan of ', result[i][1])
            optimal.append(result[i][0])

    return optimal


# Validation of the result -> GA로 얻어진 optimal individual로 이루어진 optimal list를 입력해서 가시화
def show_optimum_result(optimal):
    """
    Optimal : Set of Individuals (requires individual.seq_job
    """
    for individual in optimal:
        sequence = individual.seq_job
        env = simpy.Environment()
        scheduler = Scheduler(env, jobs_data, N_MACHINE, sequence)  # sequence = list
        scheduler.schedule()
        env.process(scheduler.evaluate())
        env.run()
        for i in range(len(scheduler.machine_list)):
            print('M%d ' % i, scheduler.machine_list[i].workingtime_log)

        del env, scheduler


"""
/***********************************************************************************************

5회 Schedule 객체 생성
generate_individual() -> 랜덤 수열 생성해서 하나의 Individual 객체 반환하는 함수

***********************************************************************************************/
"""


def generate_individual():
    sequence_ = [i for i in range(N_OPERATION)]  # sequence = list
    sequence_ = np.array(random.sample(sequence_, len(sequence_)))  # sequence = numpy array
    new = Individual(sequence_.tolist())
    return new

"""
랜덤으로 5개 스케줄 결과 생성
"""
#
# population = []
# for _ in range(50):
#     ind = generate_individual()
#     population.append(ind)
#
# for individual in population:
#     sequence = individual.seq_job
#     #print(sequence)
#     scheduler = Scheduler(jobs_data, N_MACHINE, sequence)  # sequence = list
#     # print(scheduler)
#     #
#     # print(scheduler)
#     # Gantt Chart 그리기 (Chrome 등 기본브라우저에서 열림, json 형식)
#     data = defaultdict(list)
#     # solutions = []
#     #
#     # for j in range(10):
#     #     for i in range(len(scheduler.machine_list)):
#     #         log = scheduler.machine_list[i].workingtime_log
#     #         solutions.append(log[j][0])
#
#     for i in range(len(scheduler.machine_list)):
#         print('M%d ' % i, scheduler.machine_list[i].workingtime_log)
#         # for log in scheduler.machine_list[i].workingtime_log:
#         #     solutions.append(log[0])
#
#         for j in scheduler.machine_list[i].workingtime_log:
#             temp = {'Machine': i, 'Job': j[0],
#                     'Start': j[1],
#                     'Finish': j[2]}
#             for k, v in temp.items():
#                 data[k].append(v)
#
#     data = pd.DataFrame(data)
#     data['delta'] = data['Finish'] - data['Start']
#     fig = px.timeline(data, x_start="Start", x_end="Finish", y="Machine", color="Job")
#     fig.update_yaxes(autorange="reversed")  # otherwise tasks are listed from the bottom up
#     fig.layout.xaxis.type = 'linear'
#     fig.data[0].x = data.delta.tolist()
#     fig.show()
#     print(scheduler.c_max)
#     # print(solutions, len(solutions))
#     # print('-' * 50)
#
#
#     # for i in range(len(scheduler.machine_list)):
#     #     print('M%d ' % i, scheduler.machine_list[i].workingtime_log)
#
# #
