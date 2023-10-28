import torch
import numpy as np
import pandas as pd


def NEH_sequence(env, test_input):
    if isinstance(test_input, torch.Tensor):
        processing_time = test_input.cpu().numpy()
    else:
        processing_time = np.array(test_input)
    processing_time_sum = np.sum(processing_time, axis=1)
    temp = np.argsort(processing_time_sum)[::-1]

    partial_sequence = []
    partial_processing_time = []
    for i in range(env.num_of_blocks):
        makespan_list = []
        if len(partial_sequence) == 0:
            partial_sequence.insert(0, temp[i])
            partial_processing_time.insert(0, processing_time[temp[i]])
        else:
            for pos in range(len(partial_sequence) + 1):
                partial_processing_time_temp = partial_processing_time[:]
                partial_processing_time_temp.insert(pos, processing_time[temp[i]])
                makespan_list.append(env.calculate_makespan(np.array(partial_processing_time_temp),
                                                            [j for j in range(len(partial_processing_time_temp))]).item())
            idx = np.argmin(makespan_list)
            partial_sequence.insert(int(idx), temp[i])
            partial_processing_time.insert(int(idx), processing_time[temp[i]])

    sequence = partial_sequence
    makespan = env.calculate_makespan(processing_time, sequence).item()

    return sequence, makespan


def Palmer_sequence(env, test_input):
    if isinstance(test_input, torch.Tensor):
        processing_time = test_input.cpu().numpy()
    else:
        processing_time = np.array(test_input)

    index = np.zeros(env.num_of_blocks)
    for i, processing_time_each_block in enumerate(processing_time):
        for j in range(1, env.num_of_process + 1):
            index[i] += (2 * j - env.num_of_process - 1) * processing_time_each_block[j-1] / 2

    sequence = index.argsort()[::-1]
    makespan = env.calculate_makespan(test_input, sequence).item()
    return sequence, makespan


def Campbell_sequence(env, test_input):
    if isinstance(test_input, torch.Tensor):
        processing_time = test_input.cpu().numpy()
    else:
        processing_time = np.array(test_input)
    makespan_k = []
    sequence_k = []

    for k in range(env.num_of_process - 1):
        seq = [0 for _ in range(env.num_of_blocks)]
        start = 0
        end = env.num_of_blocks - 1

        processing_time_johnson = pd.DataFrame(index=['blk_' + str(i) for i in range(env.num_of_blocks)],
                                               columns=['P1', 'P2'])
        for i in range(env.num_of_blocks):
            processing_time_johnson.iloc[i]['P1'] = sum([temp for temp in processing_time[i, :k + 1]])
            processing_time_johnson.iloc[i]['P2'] = sum([temp for temp in processing_time[i, k + 1:]])

        while len(processing_time_johnson):
            processing_time_min = np.min(processing_time_johnson)
            if processing_time_min['P1'] <= processing_time_min['P2']:
                min_idx = np.argmin(processing_time_johnson['P1'])
                if type(min_idx) == list:
                    min_idx = min_idx[0]
                seq[start] = int(processing_time_johnson.index[min_idx][4:])
                processing_time_johnson.drop(processing_time_johnson.index[min_idx], inplace=True)
                start += 1
            elif processing_time_min['P1'] > processing_time_min['P2']:
                min_idx = np.argmin(processing_time_johnson['P2'])
                if type(min_idx) == list:
                    min_idx = min_idx[0]
                seq[end] = int(processing_time_johnson.index[min_idx][4:])
                processing_time_johnson.drop(processing_time_johnson.index[min_idx], inplace=True)
                end -= 1
            else:
                min_P1_idx = np.argmin(processing_time_johnson['P1'])
                if type(min_P1_idx) == list:
                    min_P1_idx = min_P1_idx[0]
                seq[start] = int(processing_time_johnson.index[min_P1_idx][4:])
                processing_time_johnson.drop(processing_time_johnson.index[min_P1_idx], inplace=True)
                start += 1

                if len(processing_time_johnson):
                    min_P2_idx = np.argmin(processing_time_johnson['P2'])
                    if type(min_P2_idx) == list:
                        min_P2_idx = min_P2_idx[0]
                    seq[end] = int(processing_time_johnson.index[min_P2_idx][4:])
                    processing_time_johnson.drop(processing_time_johnson.index[min_P2_idx], inplace=True)
                    end -= 1

        makespan_k.append(env.calculate_makespan(test_input, seq).item())
        sequence_k.append(seq)

    best_sequence = sequence_k[np.argmin(makespan_k)]
    best_makespan = np.min(makespan_k)

    return best_sequence, best_makespan


def random_sequence(env, test_input):
    if isinstance(test_input, torch.Tensor):
        processing_time = test_input.cpu().numpy()
    else:
        processing_time = np.array(test_input)

    sequence = np.random.permutation(env.num_of_blocks)
    makespan = env.calculate_makespan(processing_time, sequence).item()
    return sequence, makespan


def SPT_sequence(env, test_input):
    if isinstance(test_input, torch.Tensor):
        processing_time = test_input.cpu().numpy()
    else:
        processing_time = np.array(test_input)

    processing_time_sum = np.sum(processing_time, axis=1)
    sequence = processing_time_sum.argsort()[::-1]
    makespan = env.calculate_makespan(processing_time, sequence).item()
    return sequence, makespan


def LPT_sequence(env, test_input):
    if isinstance(test_input, torch.Tensor):
        processing_time = test_input.cpu().numpy()
    else:
        processing_time = np.array(test_input)

    processing_time_sum = np.sum(processing_time, axis=1)
    sequence = processing_time_sum.argsort()
    makespan = env.calculate_makespan(processing_time, sequence).item()
    return sequence, makespan