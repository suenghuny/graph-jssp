import torch
import random
import numpy as np
import pandas as pd

from time import time
from datetime import datetime

from environment.env import PanelBlockShop
from environment.panelblock import *
from agent.search import *
from benchmark.heuristics import *


def test_model(env, params, data, method_list, makespan_path=None, time_path=None):
    makespan_list = []
    time_list = []

    for method in method_list:
        makespan_temp = []
        time_temp = []
        print('method: %s' % method)
        if "Ptr" in method:
            batch_size = int(method.split("-")[-1])
            for key, process_time in data.items():
                t1 = time()
                sequence, makespan = sampling(env, params, batch_size, process_time)
                t2 = time()
                t = t2 - t1
                makespan_temp.append(makespan)
                time_temp.append(t)
        elif method == "NEH":
            for key, process_time in data.items():
                t1 = time()
                sequence, makespan = NEH_sequence(env, process_time)
                t2 = time()
                t = t2 - t1
                makespan_temp.append(makespan)
                time_temp.append(t)
        elif method == "Palmer":
            for key, process_time in data.items():
                t1 = time()
                sequence, makespan = Palmer_sequence(env, process_time)
                t2 = time()
                t = t2 - t1
                makespan_temp.append(makespan)
                time_temp.append(t)
        elif method == "Campbell":
            for key, process_time in data.items():
                t1 = time()
                sequence, makespan = Campbell_sequence(env, process_time)
                t2 = time()
                t = t2 - t1
                makespan_temp.append(makespan)
                time_temp.append(t)
        elif method == "LPT":
            for key, process_time in data.items():
                t1 = time()
                sequence, makespan = LPT_sequence(env, process_time)
                t2 = time()
                t = t2 - t1
                makespan_temp.append(makespan)
                time_temp.append(t)
        elif method == "SPT":
            for key, process_time in data.items():
                t1 = time()
                sequence, makespan = SPT_sequence(env, process_time)
                t2 = time()
                t = t2 - t1
                makespan_temp.append(makespan)
                time_temp.append(t)
        elif method == "Random":
            for key, process_time in data.items():
                t1 = time()
                sequence, makespan = random_sequence(env, process_time)
                t2 = time()
                t = t2 - t1
                makespan_temp.append(makespan)
                time_temp.append(t)
        print(sum(makespan_temp) / len(makespan_temp))
        print(sum(time_temp) / len(time_temp))
        makespan_list.append(sum(makespan_temp) / len(makespan_temp))
        time_list.append(sum(time_temp) / len(time_temp))

    makespan_log = ''
    time_log = ''
    for i in range(len(method_list)):
        if i == len(method_list) - 1:
            makespan_log += '%.1f' % makespan_list[i]
            time_log += '%.1f' % time_list[i]
        else:
            makespan_log += '%.1f' % makespan_list[i] + ','
            time_log += '%.1f' % time_list[i] + ','
    makespan_log += '\n'
    time_log += '\n'

    with open(makespan_path, 'a') as f:
        f.write(makespan_log)
    with open(time_path, 'a') as f:
        f.write(time_log)


if __name__ == '__main__':

    model = "ppo"
    num_of_processes = [5]
    num_of_blocks = [125]
    batch_size = [100]
    method = ['Ptr-%d' % i for i in batch_size]
    # method = ['Ptr-%d' % i for i in batch_size] + ['NEH', 'Palmer', 'Campbell', 'LPT', 'SPT', 'Random']

    test_dir = "./result/test"

    if not os.path.exists(test_dir + "/" + model):
        os.makedirs(test_dir + "/" + model)

    date = datetime.now().strftime('%m%d_%H_%M')
    tit = ''
    for i in range(len(method)):
        if i == len(method) - 1:
            tit += method[i]
        else:
            tit += method[i] + ','
    tit += '\n'

    makespan_path = test_dir + '/' + model + '/%s_makespan.csv' % date
    with open(makespan_path, 'w') as f:
        f.write(tit)

    time_path = test_dir + '/' + model + '/%s_time.csv' % date
    with open(time_path, 'w') as f:
        f.write(tit)

    for i in num_of_processes:
        model_path = "./result/model/ppo/process_%d.pt" % i
        for j in num_of_blocks:
            data_path = "../environment/data/PBS_%d_%d.xlsx" % (i, j)
            # data = generate_block_data(num_of_process=params["num_of_process"], num_of_blocks=params["num_of_blocks"],
            #                            size=30, distribution="uniform")
            data = read_block_data(data_path)
            env = PanelBlockShop(i, j, distribution="uniform")

            params = {
                "model": model,
                "num_of_process": i,
                "num_of_blocks": j,
                "model_path": model_path,
                "n_embedding": 1024,
                "n_hidden": 512,
                "init_min": -0.08,
                "init_max": 0.08,
                "use_logit_clipping": False,
                "C": 10,
                "T": 1.5,
                "decode_type": "sampling",
                "n_glimpse": 1,
            }

            test_model(env, params, data, method, makespan_path=makespan_path, time_path=time_path)