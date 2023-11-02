import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from time import time
from datetime import datetime

from actor2 import PtrNet1
from critic import PtrNet2
from jssp2 import Scheduler
from cfg import get_cfg
import random
numbers = list(range(10))
random.shuffle(numbers)


cfg = get_cfg()
if cfg.vessl == True:
    import vessl
    vessl.init()
# torch.autograd.set_detect_anomaly(True)
# machine, procesing time

ORB10 = [
    [(9, 66), (8, 13), (0, 93), (7, 91), (6, 14), (5, 70), (3, 99), (2, 53), (4, 86), (1, 16)],
    [(8, 34), (9, 99), (0, 62), (7, 65), (5, 62), (4, 64), (6, 21), (2, 12), (3, 9), (1, 75)],
    [(9, 12), (8, 26), (7, 64), (6, 92), (4, 67), (5, 28), (3, 66), (2, 83), (1, 38), (0, 58)],
    [(0, 77), (1, 73), (3, 82), (2, 75), (6, 84), (4, 19), (5, 18), (7, 89), (8, 8), (9, 73)],
    [(0, 34), (1, 74), (7, 48), (5, 44), (4, 92), (6, 40), (3, 60), (2, 62), (8, 22), (9, 67)],
    [(9, 8),  (8, 85), (3, 58), (7, 97), (5, 92), (4, 89), (6, 75), (2, 77), (1, 95), (0, 5)],
    [(8, 52), (9, 43), (6, 5), (7, 78), (5, 12), (3, 62), (4, 21), (2, 80), (1, 60), (0, 31)],
    [(9, 81), (8, 23), (7, 23), (6, 75), (4, 78), (5, 56), (3, 51), (2, 39), (1, 53), (0, 96)],
    [(9, 79), (8, 55), (2, 88), (4, 21), (5, 83), (3, 93), (6, 47), (7, 10), (0, 63), (1, 14)],
    [(0, 43), (1, 63), (2, 83), (3, 29), (4, 52), (5, 98), (6, 54), (7, 39), (8, 33), (9, 23)]
]









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


def train_model( params, log_path=None):
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
    cri_model = PtrNet2(params).to(device)
    if params["optimizer"] == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=params["lr"])
        cri_optim = optim.Adam(cri_model.parameters(), lr=params["lr_critic"])
    elif params["optimizer"] == "RMSProp":
        act_optim = optim.RMSprop(act_model.parameters(), lr=params["lr"])
        cri_optim = optim.RMSprop(cri_model.parameters(), lr=params["lr_critic"])



    if params["is_lr_decay"]:
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],gamma=params["lr_decay"])
        cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, step_size=params["lr_decay_step_critic"],gamma=params["lr_decay"])

    mse_loss = nn.MSELoss()
    t1 = time()
    ave_makespan = 0
    min_makespans = []
    mean_makespans = []
    for s in range(epoch + 1, params["step"]):
        """
        변수별 shape 
        inputs : batch_size X number_of_blocks X number_of_process
        pred_seq : batch_size X number_of_blocks
        """
        if s % 5 == 1:
            jobs_data = []
            num_jobs = 10
            num_operation = 10
            for job in range(num_jobs):
                machine_sequence = list(range(num_jobs))
                random.shuffle(machine_sequence)
                empty = list()
                for ops in range(num_operation):
                    empty.append((machine_sequence[ops], np.random.randint(1, 100)))
                jobs_data.append(empty)
            #print(jobs_data)

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
        #rem = s% 10
        #jobs_data = datas[rem]

        if s % 20 == 1:
            num_val = 50
            act_model.init_mask_job_count(num_val)
            scheduler = Scheduler(ORB10)
            node_feature = scheduler.get_node_feature()
            node_feature = [node_feature for _ in range(num_val)]
            edge_precedence = scheduler.get_edge_index_precedence()
            edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
            edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
            heterogeneous_edges = (edge_precedence, edge_antiprecedence, edge_machine_sharing)
            heterogeneous_edges = [heterogeneous_edges for _ in range(num_val)]
            input_data = (node_feature, heterogeneous_edges)
            pred_seq, ll_old, _ = act_model(input_data, device)
            val_makespan = list()
            for sequence in pred_seq:
                scheduler = Scheduler(jobs_data)
                scheduler.run(sequence.tolist())
                makespan = scheduler.c_max
                val_makespan.append(makespan)

            print((np.min(val_makespan)/944-1)*100, (np.mean(val_makespan)/944-1)*100)
            if cfg.vessl == True:
                vessl.log(step=s, payload={'min makespan': (np.min(val_makespan)/944-1)*100})
                vessl.log(step=s, payload={'mean makespan': (np.mean(val_makespan) / 944 - 1) * 100})
            else:
                min_makespans.append((np.min(val_makespan)/944-1)*100)
                mean_makespans.append((np.mean(val_makespan) / 944 - 1) * 100)
                min_m = pd.DataFrame(min_makespans)
                mean_m = pd.DataFrame(mean_makespans)
                min_m.to_csv('min_makespan.csv')
                mean_m.to_csv('mean_makespan.csv')

            act_model.init_mask_job_count(params['batch_size'])



        act_model.block_indices = []
        scheduler = Scheduler(jobs_data)
        if params['gnn'] == True:
            node_feature = scheduler.get_node_feature()
            node_feature = [node_feature for _ in range(params['batch_size'])]
            edge_precedence = scheduler.get_edge_index_precedence()
            edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
            edge_machine_sharing = scheduler.get_machine_sharing_edge_index()


            # # Edge data
            # edge1 = edge_precedence
            # edge2 = edge_machine_sharing
            # print(edge2)
            # # Create a graph
            #
            # G = nx.Graph()
            #
            # # Add edges from edge1 and edge2
            # for e in range(len(edge1[0])):
            #     G.add_edge(edge1[0][e], edge1[1][e], weight=1)
            #
            # for e in range(len(edge2[0])):
            #     G.add_edge(edge2[0][e], edge2[1][e], weight=1)
            #
            # # for edge in edge2:
            # #     G.add_edge(edge[0], edge[1], weight=1)
            #
            # # Draw the graph
            # pos = nx.spring_layout(G)
            #
            # # Draw nodes
            # nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue")
            # print("??")
            # # Draw edges from edge1 in red and from edge2 in blue
            # nx.draw_networkx_edges(G, pos, edgelist=edge1, edge_color='red', width=2)
            # nx.draw_networkx_edges(G, pos, edgelist=edge2, edge_color='blue', width=2)
            #
            # # Draw node labels
            # nx.draw_networkx_labels(G, pos)
            #
            # # Draw edge weights
            # labels = nx.get_edge_attributes(G, 'weight')
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
            # plt.show()
            # plt.savefig("xxx.png")
            #

            heterogeneous_edges = (edge_precedence, edge_antiprecedence, edge_machine_sharing)
            heterogeneous_edges = [heterogeneous_edges for _ in range(params['batch_size'])]
            input_data = (node_feature, heterogeneous_edges)
        else:
            input_data = torch.tensor(jobs_data, dtype=torch.float).reshape(-1, 2).unsqueeze(0)
            input_data[:, :, 1] = input_data[:, :, 1].squeeze(0).squeeze(0) * 1 / input_data[:, :, 1].squeeze(0).squeeze(0).max()
            encoded_data = one_hot_encode(input_data[:, :, 0:1], 10)
            new_input_data = torch.cat([encoded_data, input_data[:, :, 1:2]], dim=-1)
            input_data = new_input_data.repeat(params["batch_size"], 1, 1)
        pred_seq, ll_old, _ = act_model(input_data, device)
        real_makespan = list()
        for sequence in pred_seq:
            scheduler = Scheduler(jobs_data)
            scheduler.run(sequence.tolist())
            makespan = - scheduler.c_max / 15
            real_makespan.append(makespan)
        ave_makespan += sum(real_makespan)/(params["batch_size"]*params["log_step"])
        if cfg.vessl == True:
            vessl.log(step=s, payload={'makespan': sum(real_makespan)/params["batch_size"]})
        # if s % 10 == 0:
        #     if act_model.T >= 0.05:
        #         act_model.T *=1
        pred_makespan = cri_model(input_data, device).unsqueeze(-1)
        adv = torch.tensor(real_makespan).detach().unsqueeze(1).to(device) - pred_makespan.detach().to(device)
        cri_loss = mse_loss(pred_makespan, torch.tensor(real_makespan, dtype = torch.float).to(device).unsqueeze(1).detach())
        cri_optim.zero_grad()
        cri_loss.backward()
        nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm=1., norm_type=2)
        cri_optim.step()
        if params["is_lr_decay"]:
            cri_lr_scheduler.step()
        #print(ll_old.shape, adv.shape)
        act_loss = -(ll_old*adv).mean()
        act_optim.zero_grad()
        act_loss.backward()
        act_optim.step()
        nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=1.0, norm_type=2)
        if params["is_lr_decay"]:
            act_lr_scheduler.step()
        ave_cri_loss += cri_loss.item()
        ave_act_loss += act_loss.item()
        # for k in range(params["iteration"]):  # K-epoch
        #     pred_makespan = cri_model(input_data, device).unsqueeze(-1)
        #     adv = torch.tensor(real_makespan).detach().unsqueeze(1).to(device) - pred_makespan.detach().to(device)
        #     cri_loss = mse_loss(pred_makespan, torch.tensor(real_makespan, dtype = torch.float).to(device).unsqueeze(1).detach())
        #     cri_optim.zero_grad()
        #     cri_loss.backward()
        #     nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm=1., norm_type=2)
        #     cri_optim.step()
        #     if params["is_lr_decay"]:
        #         cri_lr_scheduler.step()
        #     ave_cri_loss += cri_loss.item()
        #     _, ll_new, _ = act_model(input_data, device, pred_seq)  # pi(seq|inputs)
        #     ratio = torch.exp(ll_new - ll_old.detach()).unsqueeze(-1)  #
        #     surr1 = ratio * adv
        #     surr2 = torch.clamp(ratio, 1 - params["epsilon"], 1 + params["epsilon"]) * adv
        #     act_loss = -torch.min(surr1, surr2).mean()
        #     act_optim.zero_grad()
        #     act_loss.backward()
        #     act_optim.step()
        #     nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=10.0, norm_type=2)
        #     if params["is_lr_decay"]:
        #         act_lr_scheduler.step()
        #     ave_act_loss += act_loss.item()



        if s % params["log_step"] == 0:
            t2 = time()


            print('step:%d/%d, actic loss:%1.3f, crictic loss:%1.3f, L:%1.3f, %dmin%dsec' % (s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]),ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan, (t2 - t1) // 60,(t2 - t1) % 60))
            ave_makespan = 0
            if log_path is None:
                log_path = params["log_dir"] + '/ppo' + '/%s_train.csv' % date
                with open(log_path, 'w') as f:
                    f.write('step,actic loss, crictic loss, average makespan,time\n')
            else:
                with open(log_path, 'a') as f:
                    f.write('%d,%1.4f,%1.4f,%1.4f,%dmin%dsec\n' % (
                    s, ave_act_loss / ((s + 1) * params["iteration"]), ave_cri_loss / ((s + 1) * params["iteration"]),
                    ave_makespan / (s + 1),
                    (t2 - t1) // 60, (t2 - t1) % 60))
            t1 = time()

        if s % params["save_step"] == 1:
            torch.save({'epoch': s,
                        'model_state_dict_actor': act_model.state_dict(),
                        'model_state_dict_critic': cri_model.state_dict(),
                        'optimizer_state_dict_actor': act_optim.state_dict(),
                        'optimizer_state_dict_critic': cri_optim.state_dict(),
                        'ave_act_loss': ave_act_loss,
                        'ave_cri_loss': ave_cri_loss,
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
        "num_of_process":3,
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
        "layers":eval(cfg.layers),
        "lr_critic": cfg.lr_critic,
        "n_embedding": cfg.n_embedding,
        "graph_embedding_size" : cfg.graph_embedding_size
    }

    train_model(params)