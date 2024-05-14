import os
from scipy import stats
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

numbers = list(range(10))
random.shuffle(numbers)

if cfg.vessl == True:
    import vessl

    vessl.init()


opt_list = [1059, 888, 1005, 1005, 887, 1010, 397, 899, 934, 944]
orb_list = []
for i in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
    df = pd.read_excel("orb.xlsx", sheet_name=i, engine='openpyxl')
    orb_data = list()
    for row, column in df.iterrows():
        job = []
        for j in range(0, len(column.tolist()), 2):
            element = (column.tolist()[j], column.tolist()[j + 1])
            job.append(element)
        orb_data.append(job)
    orb_list.append(orb_data)


def generate_jssp_instance(num_jobs, num_operation, batch_size):
    jobs_datas = []
    for _ in range(batch_size):
        temp = []
        for job in range(num_jobs):
            machine_sequence = list(range(num_jobs))
            random.shuffle(machine_sequence)
            empty = list()
            for ops in range(num_operation):
                empty.append((machine_sequence[ops], np.random.randint(1, 100)))
            temp.append(empty)
        jobs_datas.append(temp)
    scheduler_list = list()
    for n in range(len(jobs_datas)):
        scheduler_list.append(AdaptiveScheduler(jobs_datas[n]))
    return jobs_datas, scheduler_list


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
    baseline_model = PtrNet1(params).to(device)
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

    c_max = list()
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
        b += 1


        if s % 20 == 1:
            for p in problem_list:
                num_val = 50
                scheduler_list_val = [AdaptiveScheduler(orb_list[p - 1]) for _ in range(num_val)]
                val_makespan = list()

                act_model.get_jssp_instance(scheduler_list_val)
                baseline_model.get_jssp_instance(scheduler_list_val)

                act_model.init_mask_job_count(num_val)
                baseline_model.init_mask_job_count(num_val)

                act_model.eval()
                scheduler = Scheduler(orb_list[p - 1])  # scheduler는 validation(ORB set)에 대해 수행
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
                pred_seq, ll_old, _ = act_model(input_data, device, scheduler_list=scheduler_list_val)
                for sequence in pred_seq:
                    scheduler = Scheduler(orb_list[p - 1])
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

        act_model.block_indices = []
        baseline_model.block_indices = []

        num_jobs = 10
        num_operation = 10
        if s % cfg.gen_step == 1:
            jobs_datas, scheduler_list = generate_jssp_instance(num_jobs=num_jobs, num_operation=num_operation,
                                                                batch_size=params['batch_size'])
        else:
            for scheduler in scheduler_list:
                scheduler.reset()
        act_model.get_jssp_instance(scheduler_list)

        if params['gnn'] == True:
            heterogeneous_edges = list()
            node_features = list()
            for n in range(params['batch_size']):
                scheduler = Scheduler(jobs_datas[n])
                node_feature = scheduler.get_node_feature()
                node_features.append(node_feature)
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
        act_model.train()
        pred_seq, ll_old, _ = act_model(input_data, device, scheduler_list=scheduler_list)
        real_makespan = list()
        for n in range(len(node_features)):
            sequence = pred_seq[n]
            scheduler = Scheduler(jobs_datas[n])
            makespan = -scheduler.run(sequence.tolist()) / params['reward_scaler']
            real_makespan.append(makespan)
            c_max.append(makespan)

        ave_makespan += sum(real_makespan) / (params["batch_size"] * params["log_step"])
        """

        vanila actor critic

        """
        if s == 1:
            be = torch.tensor(real_makespan).detach().unsqueeze(1).to(device)
        else:
            be = cfg.beta * be + (1 - cfg.beta) * torch.tensor(real_makespan).to(device)
        act_optim.zero_grad()
        adv = torch.tensor(real_makespan).detach().unsqueeze(1).to(
            device) - be  # torch.tensor(real_makespan_greedy).detach().unsqueeze(1).to(device)
        act_loss = -(ll_old * adv).mean()
        act_loss.backward()
        act_optim.step()
        nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=10.0, norm_type=2)
        if act_lr_scheduler.get_last_lr()[0] >= 1e-4:
            if params["is_lr_decay"]:
                # print(act_lr_scheduler.get_last_lr())
                act_lr_scheduler.step()

        # print(act_lr_scheduler.get_last_lr())

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
        "n_multi_head": cfg.n_multi_head,
        "entropy_weight": cfg.entropy_weight,
        "dot_product": cfg.dot_product
    }

    train_model(params)