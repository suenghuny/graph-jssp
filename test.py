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
from jssp2 import AdaptiveScheduler
from cfg import get_cfg

cfg = get_cfg()
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed((seed))


numbers = list(range(10))
random.shuffle(numbers)

if cfg.vessl == True:
    import vessl
    vessl.init()


# opt_list = [1059, 888, 1005, 1005, 887, 1010, 397, 899, 934, 944]
# orb_list = []
# for i in ['01','02','03','04','05','06','07','08','09','10']:
#     df = pd.read_excel("orb.xlsx", sheet_name=i)
opt_list = [3007, 3224, 3292, 3299, 3039]
orb_list = []
for i in ['71', '45']:
    df = pd.read_excel("dmu.xlsx", sheet_name=i)
    orb_data = list()
    for row, column in df.iterrows():
        job = []
        for j in range(0, len(column.tolist()), 2):
            element = (column.tolist()[j],  column.tolist()[j+1])
            job.append(element)
        orb_data.append(job)
    orb_list.append(orb_data)
    print(orb_data)

def train_model(params, log_path=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    date = datetime.now().strftime('%m%d_%H_%M')
    param_path = params["log_dir"] + '/ppo' + '/%s_%s_param.csv' % (date, "train")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))

    epoch = 0
    PATH = model_dir + "/ppo/" + "0520_08_59_step5081_act.pt"
    checkpoint = torch.load(PATH)
    act_model = PtrNet1(params).to(device)
    act_model.load_state_dict(checkpoint['model_state_dict_actor'])


    baseline_model= PtrNet1(params).to(device)
    baseline_model.load_state_dict(act_model.state_dict())
    if params["optimizer"] == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=params["lr"])
    elif params["optimizer"] == "RMSProp":
        act_optim = optim.RMSprop(act_model.parameters(), lr=params["lr"])

    if params["is_lr_decay"]:
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])

    p = 5
    num_val = 200
    scheduler_list_val = [AdaptiveScheduler(orb_list[p - 1]) for _ in range(num_val)]
    val_makespan = list()
    act_model.init_mask_job_count(num_val)
    baseline_model.init_mask_job_count(num_val)
    act_model.eval()
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
    pred_seq, ll_old, _ = act_model(input_data, device, scheduler_list = scheduler_list_val)




    for sequence in pred_seq:
        scheduler = Scheduler(orb_list[p-1])
        makespan = scheduler.run(sequence.tolist())
        val_makespan.append(makespan)


    print("ORB{}".format(p), (np.min(val_makespan) / opt_list[p - 1] - 1) * 100,
          (np.mean(val_makespan) / opt_list[p - 1] - 1) * 100, np.min(val_makespan))

        #


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

    params = {
        "num_of_process": 5,
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