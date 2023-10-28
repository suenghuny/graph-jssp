import os
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats

from benchmark.heuristics import NEH_sequence


def generate_block_data(num_of_process=6, num_of_blocks=50, size=1, distribution="lognormal"):

    if distribution == "lognormal":
        shape = [0.543, 0.525, 0.196, 0.451, 0.581, 0.432]
        scale = [2.18, 2.18, 0.518, 2.06, 1.79, 2.10]
    elif distribution == "uniform":
        loc = [0 for _ in range(num_of_process)]
        scale = [100 for _ in range(num_of_process)]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    process_time_temp = np.zeros((size * num_of_blocks, num_of_process))
    process_time = {}

    for i in range(num_of_process):
        if distribution == "lognormal":
            r = np.round(stats.lognorm.rvs(shape[i], loc=0, scale=scale[i], size=size * num_of_blocks), 1)
        elif distribution == "uniform":
            r = np.ceil(stats.uniform.rvs(loc=loc[i], scale=scale[i],size=size * num_of_blocks)).astype(np.int64)
        process_time_temp[:, i] = r
    process_time_temp = process_time_temp.reshape((size, num_of_blocks, num_of_process))

    for i in range(size):
        process_time[str(i)] = torch.FloatTensor(process_time_temp[i]).to(device)

    return process_time


def write_block_date(filepath, num_of_process=6, num_of_blocks=40, case=30, distribution="lognormal"):

    if distribution == "lognormal":
        shape = [0.543, 0.525, 0.196, 0.451, 0.581, 0.432]
        scale = [2.18, 2.18, 0.518, 2.06, 1.79, 2.10]
    elif distribution == "uniform":
        loc = [0 for _ in range(num_of_process)]
        scale = [100 for _ in range(num_of_process)]

    process_time = {}
    for i in range(case):
        process_time_temp = np.zeros((num_of_blocks, num_of_process))
        for j in range(num_of_process):
            if distribution == "lognormal":
                r = np.round(stats.lognorm.rvs(shape[j], loc=0, scale=scale[j], size=num_of_blocks), 1)
            elif distribution == "uniform":
                r = np.round(stats.uniform.rvs(loc=loc[j], scale=scale[j], size=num_of_blocks), 1)
            process_time_temp[:, j] = r
        process_time["sheet_%d" % i] = pd.DataFrame(process_time_temp)

    writer = pd.ExcelWriter(filepath + '/PBS_%d_%d.xlsx' % (num_of_process, num_of_blocks), engine='openpyxl')
    for key, value in process_time.items():
        value.to_excel(writer, sheet_name=key, index=False)
    writer.save()


def read_block_data(filepath):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    process_time = {}

    df_process_time = pd.read_excel(filepath, sheet_name=None, engine="openpyxl")
    for key, value in df_process_time.items():
        process_time[key] = torch.FloatTensor(value.to_numpy()).to(device)

    return process_time


if __name__ == "__main__":
    file_dir = "./data"

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    write_block_date(file_dir, num_of_process=5, num_of_blocks=75, case=30, distribution="uniform")
    write_block_date(file_dir, num_of_process=5, num_of_blocks=125, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=6, num_of_blocks=100, case=30, distribution="lognormal")
    # write_block_date(file_dir, num_of_process=6, num_of_blocks=200, case=30, distribution="lognormal")
    # write_block_date(file_dir, num_of_process=5, num_of_blocks=25, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=5, num_of_blocks=50, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=5, num_of_blocks=100, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=5, num_of_blocks=200, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=10, num_of_blocks=25, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=10, num_of_blocks=50, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=10, num_of_blocks=100, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=10, num_of_blocks=200, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=15, num_of_blocks=25, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=15, num_of_blocks=50, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=15, num_of_blocks=100, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=15, num_of_blocks=200, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=20, num_of_blocks=25, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=20, num_of_blocks=50, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=20, num_of_blocks=100, case=30, distribution="uniform")
    # write_block_date(file_dir, num_of_process=20, num_of_blocks=200, case=30, distribution="uniform")