import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from time import time
from datetime import datetime

from actor2 import PtrNet1
from critic import PtrNet2
from jssp import Scheduler

# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
import vessl
vessl.init()
# machine, procesing time

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
    ave_makespan = 0.0

    act_model = PtrNet1(params).to(device)
    cri_model = PtrNet2(params).to(device)
    if params["optimizer"] == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=params["lr"])
        cri_optim = optim.Adam(cri_model.parameters(), lr=params["lr"])
    elif params["optimizer"] == "RMSProp":
        act_optim = optim.RMSprop(act_model.parameters(), lr=params["lr"])
        cri_optim = optim.RMSprop(cri_model.parameters(), lr=params["lr"])

    if params["load_model"]:
        checkpoint = torch.load(params["model_dir"] + "/ppo/" + "0821_21_36_step100000_act.pt")
        act_model.load_state_dict(checkpoint['model_state_dict_actor'])
        cri_model.load_state_dict(checkpoint['model_state_dict_critic'])
        act_optim.load_state_dict(checkpoint['optimizer_state_dict_actor'])
        cri_optim.load_state_dict(checkpoint['optimizer_state_dict_critic'])
        epoch = checkpoint['epoch']
        ave_act_loss = checkpoint['ave_act_loss']
        ave_cri_loss = checkpoint['ave_cri_loss']
        ave_makespan = checkpoint['ave_makespan']
        act_model.train()
        cri_model.train()

    if params["is_lr_decay"]:
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],gamma=params["lr_decay"])
        cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, step_size=params["lr_decay_step"],gamma=params["lr_decay"])

    mse_loss = nn.MSELoss()
    t1 = time()

    for s in range(epoch + 1, params["step"]):


        #print(input_data.shape)
        """
        변수별 shape 
        inputs : batch_size X number_of_blocks X number_of_process
        pred_seq : batch_size X number_of_blocks
        """
        jobs_data = [
            [(0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)), (2, np.random.randint(1, 100)),
             (3, np.random.randint(1, 100)), (4, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
             (6, np.random.randint(1, 100)), (7, np.random.randint(1, 100)), (8, np.random.randint(1, 100)),
             (9, np.random.randint(1, 100))],
            [(0, np.random.randint(1, 100)), (2, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
             (9, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (1, np.random.randint(1, 100)),
             (6, np.random.randint(1, 100)), (5, np.random.randint(1, 100)), (7, np.random.randint(1, 100)),
             (8, np.random.randint(1, 100))],
            [(1, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (3, np.random.randint(1, 100)),
             (2, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
             (7, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
             (4, np.random.randint(1, 100))],
            [(1, np.random.randint(1, 100)), (2, np.random.randint(1, 100)), (0, np.random.randint(1, 100)),
             (4, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (8, np.random.randint(1, 100)),
             (7, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
             (5, np.random.randint(1, 100))],
            [(2, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)),
             (5, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
             (8, np.random.randint(1, 100)), (7, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
             (6, np.random.randint(1, 100))],
            [(2, np.random.randint(1, 100)), (1, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
             (3, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
             (0, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
             (7, np.random.randint(1, 100))],
            [(1, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (3, np.random.randint(1, 100)),
             (2, np.random.randint(1, 100)), (6, np.random.randint(1, 100)), (5, np.random.randint(1, 100)),
             (9, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (7, np.random.randint(1, 100)),
             (4, np.random.randint(1, 100))],
            [(2, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)),
             (5, np.random.randint(1, 100)), (4, np.random.randint(1, 100)), (6, np.random.randint(1, 100)),
             (8, np.random.randint(1, 100)), (9, np.random.randint(1, 100)), (7, np.random.randint(1, 100)),
             (3, np.random.randint(1, 100))],
            [(0, np.random.randint(1, 100)), (1, np.random.randint(1, 100)), (3, np.random.randint(1, 100)),
             (5, np.random.randint(1, 100)), (2, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
             (6, np.random.randint(1, 100)), (7, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
             (8, np.random.randint(1, 100))],
            [(1, np.random.randint(1, 100)), (0, np.random.randint(1, 100)), (2, np.random.randint(1, 100)),
             (6, np.random.randint(1, 100)), (8, np.random.randint(1, 100)), (9, np.random.randint(1, 100)),
             (5, np.random.randint(1, 100)), (3, np.random.randint(1, 100)), (4, np.random.randint(1, 100)),
             (7, np.random.randint(1, 100))]
        ]  # mach

        act_model.block_indices = []
        scheduler = Scheduler(jobs_data)
        if params['gnn'] == True:
            node_feature = scheduler.get_node_feature()
            node_feature = [node_feature for _ in range(params['batch_size'])]
            edge_precedence = scheduler.get_edge_index_precedence()
            edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
            edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
            heterogeneous_edges = (edge_precedence, edge_antiprecedence, edge_machine_sharing)
            heterogeneous_edges = [heterogeneous_edges for _ in range(params['batch_size'])]
            input_data = (node_feature, heterogeneous_edges)
        else:
            input_data = torch.tensor(jobs_data, dtype=torch.float).reshape(-1, 2).unsqueeze(0)
            input_data[:, :, 1] = input_data[:, :, 1].squeeze(0).squeeze(0) * 1 / input_data[:, :, 1].squeeze(
                0).squeeze(0).max()
            encoded_data = one_hot_encode(input_data[:, :, 0:1], 10)
            new_input_data = torch.cat([encoded_data, input_data[:, :, 1:2]], dim=-1)
            input_data = new_input_data.repeat(params["batch_size"], 1, 1)
        pred_seq, ll_old, _ = act_model(input_data, device)

        for k in range(params["iteration"]):  # K-epoch
            real_makespan = list()

            for sequence in pred_seq:
                scheduler = Scheduler(jobs_data)
                scheduler.run(sequence.tolist())
                makespan = scheduler.c_max / 150
                real_makespan.append(makespan)

            pred_makespan = cri_model(input_data, device).unsqueeze(-1)
            adv = torch.tensor(real_makespan).detach().unsqueeze(1).to(device) - pred_makespan.detach().to(device)
            cri_loss = mse_loss(pred_makespan, torch.tensor(real_makespan, dtype = torch.float).to(device).unsqueeze(1).detach())
            cri_optim.zero_grad()
            cri_loss.backward()
            nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm=1.0, norm_type=2)
            cri_optim.step()
            if params["is_lr_decay"]:
                cri_lr_scheduler.step()
            ave_cri_loss += cri_loss.item()
            _, ll_new, _ = act_model(input_data, device, pred_seq)  # pi(seq|inputs)
            ratio = torch.exp(ll_new - ll_old.detach()).unsqueeze(-1)  #



            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - params["epsilon"], 1 + params["epsilon"]) * adv

            act_loss = torch.max(surr1, surr2).mean()

            act_optim.zero_grad()
            act_loss.backward()
            act_optim.step()
            nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=1.0, norm_type=2)
            if params["is_lr_decay"]:
                act_lr_scheduler.step()
            ave_act_loss += act_loss.item()

        ave_makespan += sum(real_makespan)/params["batch_size"]

        if s % params["log_step"] == 0:
            vessl.log(step=s, payload={'makespan': ave_makespan / (s + 1)})
            t2 = time()
            print('step:%d/%d, actic loss:%1.3f, crictic loss:%1.3f, L:%1.3f, %dmin%dsec' % (s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]),ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan / (s + 1), (t2 - t1) // 60,(t2 - t1) % 60))
            t1 = time()

        # if s % params["save_step"] == 0:
        #     torch.save({'epoch': s,
        #                 'model_state_dict_actor': act_model.state_dict(),
        #                 'model_state_dict_critic': cri_model.state_dict(),
        #                 'optimizer_state_dict_actor': act_optim.state_dict(),
        #                 'optimizer_state_dict_critic': cri_optim.state_dict(),
        #                 'ave_act_loss': ave_act_loss,
        #                 'ave_cri_loss': ave_cri_loss,
        #                 'ave_makespan': ave_makespan},
        #                params["model_dir"] + '/ppo' + '/%s_step%d_act.pt' % (date, s))
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

    params = {
        "num_of_process": 1,
        "num_of_blocks": 100,
        "step": 400001,
        "log_step": 10,
        "log_dir": log_dir,
        "save_step": 1000,
        "model_dir": model_dir,
        "batch_size": 24,
        "n_hidden": 1024,
        "init_min": -0.08,
        "init_max": 0.08,
        "use_logit_clipping": True,
        "C": 10,
        "T": 1,
        "decode_type": "sampling",
        "iteration": 1,
        "epsilon": 0.2,
        "optimizer": "Adam",
        "n_glimpse": 1,
        "n_process": 3,
        "lr": 1e-4,
        "is_lr_decay": True,
        "lr_decay": 0.98,
        "num_machine": 10,
        "num_jobs": 10,
        "lr_decay_step": 2000,
        "load_model": load_model,
        "gnn": True,
        "layers":[128, 96 ,64],
        "n_embedding": 56,
        "graph_embedding_size" : 32
    }

    #env = PanelBlockShop(params["num_of_process"], params["num_of_blocks"], distribution="lognormal")
    train_model(params)