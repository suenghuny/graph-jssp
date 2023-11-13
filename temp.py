import os
from scipy import stats
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from time import time
from datetime import datetime

from actor2 import PtrNet1
from critic import PtrNet2
from jssp import Scheduler
from cfg import get_cfg

cfg = get_cfg()
import random

numbers = list(range(10))
random.shuffle(numbers)

if cfg.vessl == True:
    import vessl

    vessl.init()
# torch.autograd.set_detect_anomaly(True)
# machine, procesing time

#
# ORB1 = [
#     [(0, 72), (1, 64), (2, 55), (3, 31), (4, 53), (5, 95), (6, 11), (7, 52), (8, 6), (9, 84)],
#     [(0, 61), (3, 27), (4, 88), (2, 78), (1, 49), (5, 83), (8, 91), (6, 74), (7, 29), (9, 87)],
#     [(0, 86), (3, 32), (1, 35), (2, 37), (5, 18), (4, 48), (6, 91), (7, 52), (9, 60), (8, 30)],
#     [(0, 8), (1, 82), (4, 27), (3, 99), (6, 74), (5, 9), (2, 33), (9, 20), (7, 59), (8, 98)],
#     [(1, 50), (0, 94), (5, 43), (3, 62), (4, 55), (7, 48), (2, 5), (8, 36), (9, 47), (6, 36)],
#     [(0, 53), (6, 30), (2, 7), (3, 12), (1, 68), (8, 87), (4, 28), (9, 70), (7, 45), (5, 7)],
#     [(2, 29), (3, 96), (0, 99), (1, 14), (4, 34), (7, 14), (5, 7), (6, 76), (8, 57), (9, 76)],
#     [(2, 90), (0, 19), (3, 87), (4, 51), (1, 84), (5, 45), (9, 84), (6, 58), (7, 81), (8, 96)],
#     [(2, 97), (1, 99), (4, 93), (0, 38), (7, 13), (5, 96), (3, 40), (9, 64), (6, 32), (8, 45)],
#     [(2, 44), (0, 60), (8, 29), (3, 5), (6, 74), (1, 85), (4, 34), (7, 95), (9, 51), (5, 47)],
# ]
#
# ORB2 = [
#     [(0, 72), (1, 54), (2, 33), (3, 86), (4, 75), (5, 16), (6, 96), (7, 7), (8, 99), (9, 76)],
#     [(0, 16), (3, 88), (4, 48), (8, 52), (9, 60), (6, 29), (7, 18), (5, 89), (2, 80), (1, 76)],
#     [(0, 47), (7, 11), (3, 14), (2, 56), (6, 16), (4, 83), (1, 10), (5, 61), (8, 24), (9, 58)],
#     [(0, 49), (1, 31), (3, 17), (8, 50), (5, 63), (2, 35), (4, 65), (7, 23), (6, 50), (9, 29)],
#     [(0, 55), (6, 6), (1, 28), (3, 96), (5, 86), (2, 99), (9, 14), (7, 70), (8, 64), (4, 24)],
#     [(4, 46), (0, 23), (6, 70), (8, 19), (2, 54), (3, 22), (9, 85), (7, 87), (5, 79), (1, 93)],
#     [(4, 76), (3, 60), (0, 76), (9, 98), (2, 76), (1, 50), (8, 86), (7, 14), (6, 27), (5, 57)],
#     [(4, 93), (6, 27), (9, 57), (3, 87), (8, 86), (2, 54), (7, 24), (5, 49), (0, 20), (1, 47)],
#     [(2, 28), (6, 11), (8, 78), (7, 85), (4, 63), (9, 81), (3, 10), (1, 9), (5, 46), (0, 32)],
#     [(2, 22), (9, 76), (5, 89), (8, 13), (6, 88), (3, 10), (7, 75), (4, 98), (1, 78), (0, 17)],
# ]
#
# ORB3 = [
#     [(0, 96), (1, 69), (2, 25), (3, 5), (4, 55), (5, 15), (6, 88), (7, 11), (8, 17), (9, 82)],
#     [(0, 11), (1, 48), (2, 67), (3, 38), (4, 18), (5, 92), (6, 62), (7, 24), (8, 81), (9, 96)],
#     [(2, 67), (1, 63), (0, 93), (4, 85), (3, 25), (5, 72), (6, 51), (7, 81), (8, 58), (9, 15)],
#     [(2, 30), (1, 35), (0, 27), (4, 82), (3, 44), (5, 49), (6, 25), (7, 92), (8, 77), (9, 28)],
#     [(1, 53), (0, 83), (4, 73), (3, 26), (2, 77), (6, 33), (5, 92), (9, 99), (8, 38), (7, 38)],
#     [(1, 20), (0, 44), (4, 81), (3, 88), (2, 66), (6, 70), (5, 91), (9, 37), (8, 55), (7, 96)],
#     [(1, 21), (2, 93), (4, 22), (0, 56), (3, 34), (6, 40), (7, 53), (9, 46), (5, 29), (8, 63)],
#     [(1, 32), (2, 63), (4, 36), (0, 26), (3, 17), (5, 85), (6, 82), (7, 15), (8, 55), (9, 16)],
#     [(0, 73), (2, 46), (3, 89), (4, 24), (1, 99), (6, 92), (7, 7), (9, 51), (5, 19), (8, 14)],
#     [(0, 52), (2, 20), (3, 70), (4, 98), (1, 23), (5, 15), (7, 81), (8, 71), (9, 24), (6, 81)]
# ]
#
# ORB4 = [
#     [(0, 8), (1, 10), (2, 35), (3, 44), (4, 15), (5, 92), (6, 70), (7, 89), (8, 50), (9, 12)],
#     [(0, 63), (8, 39), (3, 80), (5, 22), (2, 88), (1, 39), (9, 85), (6, 27), (7, 74), (4, 69)],
#     [(0, 52), (6, 22), (1, 33), (3, 68), (8, 27), (2, 68), (5, 25), (4, 34), (7, 24), (9, 84)],
#     [(0, 31), (1, 85), (4, 55), (8, 80), (5, 58), (7, 11), (6, 69), (9, 56), (3, 73), (2, 25)],
#     [(0, 97), (5, 98), (9, 87), (8, 47), (7, 77), (4, 90), (3, 98), (2, 80), (1, 39), (6, 40)],
#     [(1, 97), (5, 68), (0, 44), (9, 67), (2, 44), (8, 85), (3, 78), (6, 90), (7, 33), (4, 81)],
#     [(0, 34), (3, 76), (8, 48), (7, 61), (9, 11), (2, 36), (4, 33), (6, 98), (1, 7), (5, 44)],
#     [(0, 44), (9, 5), (4, 85), (1, 51), (5, 58), (7, 79), (2, 95), (6, 48), (3, 86), (8, 73)],
#     [(0, 24), (1, 63), (9, 48), (7, 77), (8, 73), (6, 74), (4, 63), (5, 17), (2, 93), (3, 84)],
#     [(0, 51), (2, 5), (4, 40), (9, 60), (1, 46), (5, 58), (8, 54), (3, 72), (6, 29), (7, 94)]
# ]
#
# ORB5 = [
#     [(9, 11), (8, 93), (0, 48), (7, 76), (6, 13), (5, 71), (3, 59), (2, 90), (4, 10), (1, 65)],
#     [(8, 52), (9, 76), (0, 84), (7, 73), (5, 56), (4, 10), (6, 26), (2, 43), (3, 39), (1, 49)],
#     [(9, 28), (8, 44), (7, 26), (6, 66), (4, 68), (5, 74), (3, 27), (2, 14), (1, 6), (0, 21)],
#     [(0, 18), (1, 58), (3, 62), (2, 46), (6, 25), (4, 6), (5, 60), (7, 28), (8, 80), (9, 30)],
#     [(0, 78), (1, 47), (7, 29), (5, 16), (4, 29), (6, 57), (3, 78), (2, 87), (8, 39), (9, 73)],
#     [(9, 66), (8, 51), (3, 12), (7, 64), (5, 67), (4, 15), (6, 66), (2, 26), (1, 20), (0, 98)],
#     [(8, 23), (9, 76), (6, 45), (7, 75), (5, 24), (3, 18), (4, 83), (2, 15), (1, 88), (0, 17)],
#     [(9, 56), (8, 83), (7, 80), (6, 16), (4, 31), (5, 93), (3, 30), (2, 29), (1, 66), (0, 28)],
#     [(9, 79), (8, 69), (2, 82), (4, 16), (5, 62), (3, 41), (6, 91), (7, 35), (0, 34), (1, 75)],
#     [(0, 5), (1, 19), (2, 20), (3, 12), (4, 94), (5, 60), (6, 99), (7, 31), (8, 96), (9, 63)]
# ]
#
# ORB6 = [
#     [(0, 99), (1, 74), (2, 49), (3, 67), (4, 17), (5, 7), (6, 9), (7, 39), (8, 35), (9, 49)],
#     [(0, 49), (3, 67), (4, 82), (2, 92), (1, 62), (5, 84), (8, 45), (6, 30), (7, 42), (9, 71)],
#     [(0, 26), (3, 33), (1, 82), (2, 98), (5, 83), (4, 16), (6, 64), (7, 65), (9, 36), (8, 77)],
#     [(0, 41), (1, 62), (4, 73), (3, 94), (6, 51), (5, 46), (2, 55), (9, 31), (7, 64), (8, 46)],
#     [(1, 68), (0, 26), (5, 50), (3, 46), (4, 25), (7, 88), (2, 6), (8, 13), (9, 98), (6, 84)],
#     [(0, 24), (6, 80), (2, 91), (3, 55), (1, 48), (8, 99), (4, 72), (9, 91), (7, 84), (5, 12)],
#     [(2, 16), (3, 13), (0, 9), (1, 58), (4, 23), (7, 85), (5, 36), (6, 89), (8, 71), (9, 41)],
#     [(2, 54), (0, 41), (3, 38), (4, 53), (1, 11), (5, 74), (9, 88), (6, 46), (7, 41), (8, 65)],
#     [(2, 53), (1, 50), (4, 40), (0, 90), (7, 7), (5, 80), (3, 57), (9, 60), (6, 91), (8, 47)],
#     [(2, 45), (0, 59), (8, 81), (3, 99), (6, 71), (1, 19), (4, 75), (7, 77), (9, 94), (5, 95)]
# ]
#
# ORB7 = [
#     [(0, 32), (1, 14), (2, 15), (3, 37), (4, 18), (5, 43), (6, 19), (7, 27), (8, 28), (9, 31)],
#     [(0, 8), (3, 12), (4, 49), (8, 24), (9, 52), (6, 19), (7, 23), (5, 19), (2, 17), (1, 32)],
#     [(0, 25), (7, 19), (3, 27), (2, 45), (6, 21), (4, 15), (1, 13), (5, 16), (8, 43), (9, 19)],
#     [(0, 24), (1, 18), (3, 41), (8, 29), (5, 14), (2, 17), (4, 23), (7, 15), (6, 18), (9, 23)],
#     [(0, 27), (6, 29), (1, 39), (3, 21), (5, 15), (2, 15), (9, 25), (7, 26), (8, 44), (4, 20)],
#     [(4, 17), (0, 15), (6, 51), (8, 17), (2, 46), (3, 16), (9, 33), (7, 25), (5, 30), (1, 25)],
#     [(4, 15), (3, 31), (0, 25), (9, 12), (2, 13), (1, 51), (8, 19), (7, 21), (6, 12), (5, 26)],
#     [(4, 8), (6, 29), (9, 25), (3, 15), (8, 17), (2, 22), (7, 32), (5, 20), (0, 11), (1, 28)],
#     [(2, 41), (6, 10), (8, 32), (7, 5), (4, 21), (9, 59), (3, 26), (1, 10), (5, 16), (0, 29)],
#     [(2, 20), (9, 7), (5, 44), (8, 22), (6, 33), (3, 25), (7, 29), (4, 12), (1, 14), (0, 0)]
# ]
#
# ORB8 = [
#     [(0, 55), (1, 74), (2, 45), (3, 23), (4, 76), (5, 19), (6, 18), (7, 61), (8, 44), (9, 11)],
#     [(0, 63), (1, 43), (2, 51), (3, 18), (4, 42), (7, 11), (6, 29), (5, 52), (9, 29), (8, 88)],
#     [(2, 88), (1, 31), (0, 47), (4, 10), (3, 62), (5, 60), (6, 58), (7, 29), (8, 52), (9, 92)],
#     [(2, 16), (1, 71), (0, 55), (4, 55), (3, 9), (7, 49), (6, 83), (5, 54), (9, 7), (8, 57)],
#     [(1, 7), (0, 41), (4, 92), (3, 94), (2, 46), (6, 79), (5, 34), (9, 38), (8, 8), (7, 18)],
#     [(1, 25), (0, 5), (4, 89), (3, 94), (2, 14), (6, 94), (5, 20), (9, 23), (8, 44), (7, 39)],
#     [(1, 24), (2, 21), (4, 47), (0, 40), (3, 94), (6, 71), (7, 89), (9, 75), (5, 97), (8, 15)],
#     [(1, 5), (2, 7), (4, 74), (0, 28), (3, 72), (5, 61), (7, 9), (8, 53), (9, 32), (6, 97)],
#     [(0, 34), (2, 52), (3, 37), (4, 6), (1, 94), (6, 6), (7, 56), (9, 41), (5, 5), (8, 16)],
#     [(0, 77), (2, 74), (3, 82), (4, 10), (1, 29), (5, 15), (7, 51), (8, 65), (9, 37), (6, 21)]
# ]
#
# ORB9 = [
#     [(0, 36), (1, 96), (2, 86), (3, 7), (4, 20), (5, 9), (6, 39), (7, 79), (8, 82), (9, 24)],
#     [(0, 16), (8, 95), (3, 67), (5, 63), (2, 87), (1, 24), (9, 62), (6, 49), (7, 92), (4, 16)],
#     [(0, 65), (6, 71), (1, 9), (3, 67), (8, 70), (2, 48), (5, 49), (4, 66), (7, 5), (9, 96)],
#     [(0, 50), (1, 31), (4, 6), (8, 13), (5, 98), (7, 97), (6, 93), (9, 30), (3, 34), (2, 83)],
#     [(0, 99), (5, 7), (9, 55), (8, 78), (7, 68), (4, 81), (3, 90), (2, 75), (1, 66), (6, 40)],
#     [(1, 42), (5, 11), (0, 5), (9, 39), (2, 10), (8, 30), (3, 39), (6, 50), (7, 20), (4, 51)],
#     [(0, 38), (3, 68), (8, 86), (7, 77), (9, 32), (2, 89), (4, 37), (6, 53), (1, 43), (5, 89)],
#     [(0, 19), (9, 11), (4, 37), (1, 41), (5, 72), (7, 7), (2, 52), (6, 31), (3, 68), (8, 10)],
#     [(0, 83), (1, 21), (9, 23), (7, 87), (8, 58), (6, 89), (4, 74), (5, 29), (2, 74), (3, 23)],
#     [(0, 44), (2, 57), (4, 69), (9, 50), (1, 65), (5, 69), (8, 60), (3, 58), (6, 89), (7, 13)]
# ]
#
# ORB10 = [
#     [(9, 66), (8, 13), (0, 93), (7, 91), (6, 14), (5, 70), (3, 99), (2, 53), (4, 86), (1, 16)],
#     [(8, 34), (9, 99), (0, 62), (7, 65), (5, 62), (4, 64), (6, 21), (2, 12), (3, 9), (1, 75)],
#     [(9, 12), (8, 26), (7, 64), (6, 92), (4, 67), (5, 28), (3, 66), (2, 83), (1, 38), (0, 58)],
#     [(0, 77), (1, 73), (3, 82), (2, 75), (6, 84), (4, 19), (5, 18), (7, 89), (8, 8), (9, 73)],
#     [(0, 34), (1, 74), (7, 48), (5, 44), (4, 92), (6, 40), (3, 60), (2, 62), (8, 22), (9, 67)],
#     [(9, 8), (8, 85), (3, 58), (7, 97), (5, 92), (4, 89), (6, 75), (2, 77), (1, 95), (0, 5)],
#     [(8, 52), (9, 43), (6, 5), (7, 78), (5, 12), (3, 62), (4, 21), (2, 80), (1, 60), (0, 31)],
#     [(9, 81), (8, 23), (7, 23), (6, 75), (4, 78), (5, 56), (3, 51), (2, 39), (1, 53), (0, 96)],
#     [(9, 79), (8, 55), (2, 88), (4, 21), (5, 83), (3, 93), (6, 47), (7, 10), (0, 63), (1, 14)],
#     [(0, 43), (1, 63), (2, 83), (3, 29), (4, 52), (5, 98), (6, 54), (7, 39), (8, 33), (9, 23)]
# ]

opt_list = [1059, 888, 1005, 1005, 887, 1010, 397, 899, 934, 944]
orb_list = []
for i in ['01','02','03','04','05','06','07','08','09','10']:
    df = pd.read_excel("orb.xlsx", sheet_name=i)
    orb_data = list()
    for row, column in df.iterrows():
        job = []
        for j in range(0, len(column.tolist()), 2):
            element = (column.tolist()[j],  column.tolist()[j+1])
            job.append(element)
        orb_data.append(job)
    orb_list.append(orb_data)
    print(orb_data)
        #print(column)
#pd.DataFrame()
# datas = [[
#     [(0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)), (2, np.random.randint(1, 100)),
#      (3, np.random.randint(1, 100)), (4, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
#      (6, np.random.randint(1, 100)), (7, np.random.randint(1, 100)), (8, np.random.randint(1, 100)),
#      (9, np.random.randint(1, 100))],
#     [(0, np.random.randint(1, 100)), (2, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
#      (9, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (1, np.random.randint(1, 100)),
#      (6, np.random.randint(1, 100)), (5, np.random.randint(1, 100)), (7, np.random.randint(1, 100)),
#      (8, np.random.randint(1, 100))],
#     [(1, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (3, np.random.randint(1, 100)),
#      (2, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
#      (7, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
#      (4, np.random.randint(1, 100))],
#     [(1, np.random.randint(1, 100)), (2, np.random.randint(1, 100)), (0, np.random.randint(1, 100)),
#      (4, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (8, np.random.randint(1, 100)),
#      (7, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
#      (5, np.random.randint(1, 100))],
#     [(2, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)),
#      (5, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
#      (8, np.random.randint(1, 100)), (7, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
#      (6, np.random.randint(1, 100))],
#     [(2, np.random.randint(1, 100)), (1, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
#      (3, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
#      (0, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
#      (7, np.random.randint(1, 100))],
#     [(1, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (3, np.random.randint(1, 100)),
#      (2, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
#      (9, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (7, np.random.randint(1, 100)),
#      (4, np.random.randint(1, 100))],
#     [(2, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)),
#      (5, np.random.randint(1, 100)), (4, np.random.randint(1, 100)), (6, np.random.randint(1, 100)),
#      (8, np.random.randint(1, 100)), (9, np.random.randint(1, 100)), (7, np.random.randint(1, 100)),
#      (3, np.random.randint(1, 100))],
#     [(0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)), (3, np.random.randint(1, 100)),
#      (5, np.random.randint(1, 100)), (2, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
#      (6, np.random.randint(1, 100)), (7, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
#      (8, np.random.randint(1, 100))],
#     [(1, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (2, np.random.randint(1, 100)),
#      (6, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
#      (5, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
#      (7, np.random.randint(1, 100))]
# ] for _ in range(10)]


def train_model(params, log_path=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    date = datetime.now().strftime('%m%d_%H_%M')
    param_path = params["log_dir"] + '/ppo' + '/%s_%s_param.csv' % (date, "train")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))

    epoch = 0
    ave_act_loss = 0.0
    ave_cri_loss = 0.0

    act_model = PtrNet1(params).to(device)
    baseline_model= PtrNet1(params).to(device)
    baseline_model.load_state_dict(act_model.state_dict())
    if params["optimizer"] == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=params["lr"])
    elif params["optimizer"] == "RMSProp":
        act_optim = optim.RMSprop(act_model.parameters(), lr=params["lr"])

    if params["is_lr_decay"]:
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])


    mse_loss = nn.MSELoss()
    t1 = time()
    ave_makespan = 0
    min_makespans = []
    mean_makespans = []

    c_max =list()
    c_max_g = list()
    baseline_update = 30
    b = 0
    for s in range(epoch + 1, params["step"]):
        problem_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        """
        변수별 shape 
        inputs : batch_size X number_of_blocks X number_of_process
        pred_seq : batch_size X number_of_blocks
        """
        b +=1
        if s % 1 == 0:
            jobs_datas = []
            num_jobs = 10
            num_operation = 10
            for _ in range(params['batch_size']):

                # for j in range(num_jobs):
                #     temp.append(eval('ORB{}'.format(np.random.choice(problem_list)))[j])
                #temp = eval('ORB{}'.format(np.random.choice(problem_list)))
                # print(temp)
                #jobs_datas.append(temp)
                temp = []
                for job in range(num_jobs):
                    machine_sequence = list(range(num_jobs))
                    random.shuffle(machine_sequence)
                    empty = list()
                    for ops in range(num_operation):
                        empty.append((machine_sequence[ops], np.random.randint(1, 100)))
                    temp.append(empty)
                jobs_datas.append(temp)
            # print(jobs_data)
#
        # if s % 20 == 1:
        #     jobs_data = [
        #         [(0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)), (2, np.random.randint(1, 100)),
        #          (3, np.random.randint(1, 100)), (4, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
        #          (6, np.random.randint(1, 100)), (7, np.random.randint(1, 100)), (8, np.random.randint(1, 100)),
        #          (9, np.random.randint(1, 100))],
        #         [(0, np.random.randint(1, 100)), (2, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
        #          (9, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (1, np.random.randint(1, 100)),
        #          (6, np.random.randint(1, 100)), (5, np.random.randint(1, 100)), (7, np.random.randint(1, 100)),
        #          (8, np.random.randint(1, 100))],
        #         [(1, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (3, np.random.randint(1, 100)),
        #          (2, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
        #          (7, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
        #          (4, np.random.randint(1, 100))],
        #         [(1, np.random.randint(1, 100)), (2, np.random.randint(1, 100)), (0, np.random.randint(1, 100)),
        #          (4, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (8, np.random.randint(1, 100)),
        #          (7, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
        #          (5, np.random.randint(1, 100))],
        #         [(2, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)),
        #          (5, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
        #          (8, np.random.randint(1, 100)), (7, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
        #          (6, np.random.randint(1, 100))],
        #         [(2, np.random.randint(1, 100)), (1, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
        #          (3, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
        #          (0, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
        #          (7, np.random.randint(1, 100))],
        #         [(1, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (3, np.random.randint(1, 100)),
        #          (2, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
        #          (9, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (7, np.random.randint(1, 100)),
        #          (4, np.random.randint(1, 100))],
        #         [(2, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)),
        #          (5, np.random.randint(1, 100)), (4, np.random.randint(1, 100)), (6, np.random.randint(1, 100)),
        #          (8, np.random.randint(1, 100)), (9, np.random.randint(1, 100)), (7, np.random.randint(1, 100)),
        #          (3, np.random.randint(1, 100))],
        #         [(0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)), (3, np.random.randint(1, 100)),
        #          (5, np.random.randint(1, 100)), (2, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
        #          (6, np.random.randint(1, 100)), (7, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
        #          (8, np.random.randint(1, 100))],
        #         [(1, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (2, np.random.randint(1, 100)),
        #          (6, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
        #          (5, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
        #          (7, np.random.randint(1, 100))]
        #     ]  # mach
        # rem = s% 10
        # jobs_data = datas[rem]

        if s % 20 == 1:
            for p in problem_list:
                num_val = 50
                val_makespan = list()
                act_model.init_mask_job_count(num_val)
                baseline_model.init_mask_job_count(num_val)
                scheduler = Scheduler(orb_list[p-1])
                node_feature = scheduler.get_node_feature()
                node_feature = [node_feature for _ in range(num_val)]
                edge_precedence = scheduler.get_edge_index_precedence()
                edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
                edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
                edge_fcn = scheduler.get_fully_connected_edge_index()
                if cfg.fully_connected == True:
                    heterogeneous_edges = (edge_precedence, edge_antiprecedence, edge_machine_sharing, edge_fcn)
                else:
                    heterogeneous_edges = (edge_precedence, edge_antiprecedence, edge_machine_sharing)
                heterogeneous_edges = [heterogeneous_edges for _ in range(num_val)]
                input_data = (node_feature, heterogeneous_edges)
                pred_seq, ll_old, _ = act_model(input_data, device)




                for sequence in pred_seq:
                    scheduler = Scheduler(orb_list[p-1])
                    makespan = scheduler.run(sequence.tolist())
                    val_makespan.append(makespan)


                print("ORB{}".format(p), (np.min(val_makespan) / opt_list[p - 1] - 1) * 100,
                      (np.mean(val_makespan) / opt_list[p - 1] - 1) * 100, np.min(val_makespan))
                if cfg.vessl == True:
                    vessl.log(step=s, payload={
                        'min makespan_{}'.format('ORB' + str(p)): (np.min(val_makespan) / opt_list[p - 1] - 1) * 100})
                    vessl.log(step=s, payload={
                        'mean makespan_{}'.format('ORB' + str(p)): (np.mean(val_makespan) / opt_list[p - 1] - 1) * 100})
                else:
                    min_makespans.append((np.min(val_makespan) / 944 - 1) * 100)
                    mean_makespans.append((np.mean(val_makespan) / 944 - 1) * 100)
                    min_m = pd.DataFrame(min_makespans)
                    mean_m = pd.DataFrame(mean_makespans)
                    min_m.to_csv('min_makespan.csv')
                    mean_m.to_csv('mean_makespan.csv')

                act_model.init_mask_job_count(params['batch_size'])
                baseline_model.init_mask_job_count(params['batch_size'])
                #baseline_model.init_mask_job_count(num_val)

        act_model.block_indices = []
        baseline_model.block_indices = []

        if params['gnn'] == True:
            heterogeneous_edges = list()
            node_features = list()
            for n in range(params['batch_size']):
                scheduler = Scheduler(jobs_datas[n])
                node_feature = scheduler.get_node_feature()
                node_features.append(node_feature)
                # node_feature = [node_feature for _ in range(params['batch_size'])]
                edge_precedence = scheduler.get_edge_index_precedence()
                edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
                edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
                edge_fcn = scheduler.get_fully_connected_edge_index()
                if cfg.fully_connected == True:
                    heterogeneous_edge = (edge_precedence, edge_antiprecedence, edge_machine_sharing, edge_fcn)
                else:
                    heterogeneous_edge = (edge_precedence, edge_antiprecedence, edge_machine_sharing)
                heterogeneous_edges.append(heterogeneous_edge)
            input_data = (node_features, heterogeneous_edges)

        pred_seq, ll_old, _ = act_model(input_data, device)
        real_makespan = list()
        for n in range(len(node_features)):
            sequence = pred_seq[n]
            scheduler = Scheduler(jobs_datas[n])
            makespan = scheduler.run(sequence.tolist()) / params['reward_scaler']
            real_makespan.append(makespan)
            c_max.append(makespan)



        baseline_model.block_indices = []
        baseline_model.eval()
        pred_seq_greedy, _, _ = baseline_model(input_data, device, greedy=True)
        real_makespan_greedy = list()
        for sequence_g in pred_seq_greedy:
            scheduler = Scheduler(jobs_datas[n])
            makespan = scheduler.run(sequence_g.tolist()) / params['reward_scaler']
            real_makespan_greedy.append(makespan)
            c_max_g.append(makespan)

        ave_makespan += sum(real_makespan) / (params["batch_size"] * params["log_step"])
        """
        vanila actor critic
        """
        if cfg.ppo == False:
            act_optim.zero_grad()
            adv = torch.tensor(real_makespan).detach().unsqueeze(1).to(device) - torch.tensor(real_makespan_greedy).detach().unsqueeze(1).to(device)
            act_loss = (ll_old * adv).mean()
            act_loss.backward()
            act_optim.step()
            nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=10.0, norm_type=2)
            if act_lr_scheduler.get_last_lr()[0] >= 1e-4:
                if params["is_lr_decay"]:
                    # print(act_lr_scheduler.get_last_lr())
                    act_lr_scheduler.step()

            #print(act_lr_scheduler.get_last_lr())
            if s % cfg.interval== 0:
                if stats.ttest_rel(c_max, c_max_g)[1]<0.05:
                    c_max = list()
                    c_max_g = list()
                    baseline_model.load_state_dict(act_model.state_dict())

                else:
                    c_max = list()
                    c_max_g = list()
            ave_act_loss += act_loss.item()

        """
        vanila actor critic

        """

        #

        if s % params["log_step"] == 0:
            t2 = time()

            print('step:%d/%d, actic loss:%1.3f, crictic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
            s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]),
            ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan, (t2 - t1) // 60, (t2 - t1) % 60))
            ave_makespan = 0
            if log_path is None:
                log_path = params["log_dir"] + '/ppo' + '/%s_train.csv' % date
                with open(log_path, 'w') as f:
                    f.write('step,actic loss, crictic loss, average makespan,time\n')
            else:
                with open(log_path, 'a') as f:
                    f.write('%d,%1.4f,%1.4f,%1.4f,%dmin%dsec\n' % (
                        s, ave_act_loss / ((s + 1) * params["iteration"]),
                        ave_cri_loss / ((s + 1) * params["iteration"]),
                        ave_makespan / (s + 1),
                        (t2 - t1) // 60, (t2 - t1) % 60))
            t1 = time()

        if s % params["save_step"] == 1:
            torch.save({'epoch': s,
                        'model_state_dict_actor': act_model.state_dict(),
                        'optimizer_state_dict_actor': act_optim.state_dict(),
                        'ave_act_loss': ave_act_loss,
                        'ave_cri_loss': 0,
                        'ave_makespan': ave_makespan},
                       params["model_dir"] + '/ppo' + '/%s_step%d_act.pt' % (date, s))
        #     print('save model...')


def one_hot_encode(tensor, n_classes):
    original_shape = tensor.shape
    tensor = tensor.long().view(-1)
    one_hot = torch.zeros(tensor.shape[0], n_classes).to(tensor.device)
    one_hot.scatter_(1, tensor.view(-1, 1), 1)
    one_hot = one_hot.view(*original_shape[:-1], n_classes)
    return one_hot


if __name__ == '__main__':

    load_model = False

    log_dir = "./result/log"
    if not os.path.exists(log_dir + "/ppo"):
        os.makedirs(log_dir + "/ppo")

    model_dir = "./result/model"
    if not os.path.exists(model_dir + "/ppo"):
        os.makedirs(model_dir + "/ppo")

    # parser.add_argument("--vessl", type=bool, default=False, help="vessl AI 사용여부")
    # parser.add_argument("--step", type=int, default=400001, help="")
    # parser.add_argument("--save_step", type=int, default=10, help="")
    # parser.add_argument("--batch_size", type=int, default=24, help="")
    # parser.add_argument("--n_hidden", type=int, default=1024, help="")
    # parser.add_argument("--C", type=float, default=10, help="")
    # parser.add_argument("--T", type=int, default=1, help="")
    # parser.add_argument("--iteration", type=int, default=1, help="")
    # parser.add_argument("--epsilon", type=float, default=0.18, help="")
    # parser.add_argument("--n_glimpse", type=int, default=2, help="")
    # parser.add_argument("--n_process", type=int, default=3, help="")
    # parser.add_argument("--lr", type=float, default=1.2e-4, help="")
    # parser.add_argument("--lr_decay", type=float, default=0.98, help="")
    # parser.add_argument("--lr_decay_step", type=int, default=30000, help="")
    # parser.add_argument("--layers", type=str, default="[128, 108 ,96]", help="")
    # parser.add_argument("--n_embedding", type=int, default=128, help="")
    # parser.add_argument("--graph_embedding_size", type=int, default=64, help="")

    params = {
        "num_of_process": 6,
        "num_of_blocks": 100,
        "step": cfg.step,
        "log_step": cfg.log_step,
        "log_dir": log_dir,
        "save_step": cfg.save_step,
        "model_dir": model_dir,
        "batch_size": cfg.batch_size,
        "n_hidden": cfg.n_hidden,
        "init_min": -0.08,
        "init_max": 0.08,
        "use_logit_clipping": True,
        "C": cfg.C,
        "T": cfg.T,
        "decode_type": "sampling",
        "iteration": cfg.iteration,
        "epsilon": cfg.epsilon,
        "optimizer": "Adam",
        "n_glimpse": cfg.n_glimpse,
        "n_process": cfg.n_process,
        "lr": cfg.lr,
        "is_lr_decay": True,
        "lr_decay": cfg.lr_decay,

        "num_machine": 10,
        "num_jobs": 10,
        "lr_decay_step": cfg.lr_decay_step,
        "lr_decay_step_critic": cfg.lr_decay_step_critic,
        "load_model": load_model,
        "gnn": True,
        "layers": eval(cfg.layers),
        "lr_critic": cfg.lr_critic,
        "n_embedding": cfg.n_embedding,
        "graph_embedding_size": cfg.graph_embedding_size,
        "reward_scaler": cfg.reward_scaler,
        "n_multi_head":cfg.n_multi_head,
        "entropy_weight": cfg.entropy_weight,
        "dot_product":cfg.dot_product
    }

    train_model(params)