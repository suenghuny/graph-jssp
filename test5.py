import pandas as pd
import torch
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import wandb
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from time import time
from datetime import datetime
from actor_VAE2 import PtrNet1
from jssp2 import AdaptiveScheduler
from cfg import get_cfg
import random

cfg = get_cfg()
baseline_reset = cfg.baseline_reset
###

if cfg.vessl == True:
    import vessl
    import os

    vessl.init()
    output_dir = "/output/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Example usage:
set_seed(int(os.environ.get("seed", 30)))  # 30 했었음
opt_list = [1059, 888, 1005, 1005, 887, 1010, 397, 899, 934, 944]
orb_list = []
for i in ["21", "22"]:
    df = pd.read_excel("ta.xlsx", sheet_name=i, engine='openpyxl')
    orb_data = list()  #
    for row, column in df.iterrows():
        job = []
        for j in range(0, len(column.tolist()), 2):
            element = (column.tolist()[j], column.tolist()[j + 1])
            job.append(element)
        orb_data.append(job)
    orb_list.append(orb_data)




def heuristic_eval(p):  # 불필요(현재로써는)
    scheduler = AdaptiveScheduler(orb_list[p - 1])
    makespan_heu = scheduler.heuristic_run()
    return makespan_heu


def evaluation(act_model, baseline_model, p, eval_number, device, upperbound=None):
    scheduler_list_val = [AdaptiveScheduler(orb_list[p - 1]) for _ in range(eval_number)]
    val_makespan = list()
    act_model.get_jssp_instance(scheduler_list_val)
    baseline_model.get_jssp_instance(scheduler_list_val)

    act_model.eval()

    scheduler = AdaptiveScheduler(orb_list[p - 1])  # scheduler는 validation(ORB set)에 대해 수행

    num_job = scheduler.num_job
    num_machine = scheduler.num_mc
    # print(num_job, num_machine)

    node_feature = scheduler.get_node_feature()
    node_feature = [node_feature for _ in range(int(eval_number))]
    edge_precedence = scheduler.get_edge_index_precedence()
    edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
    edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
    heterogeneous_edges = (edge_precedence, edge_antiprecedence, edge_machine_sharing)
    heterogeneous_edges = [heterogeneous_edges for _ in range(eval_number)]
    input_data = (node_feature, heterogeneous_edges)
    pred_seq, ll_old, _, _, _, _, _ = act_model(input_data,
                                    device,
                                    scheduler_list=scheduler_list_val,
                                    num_machine=num_machine,
                                    num_job=num_job,
                                    upperbound=upperbound,
                                    train = False)
    for sequence in pred_seq:
        scheduler = AdaptiveScheduler(orb_list[p - 1])
        makespan = scheduler.run(sequence.tolist())
        val_makespan.append(makespan)
    # print("크크크", val_makespan)
    return np.min(val_makespan), np.mean(val_makespan)


def train_model(params, log_path=None):
    device = torch.device(cfg.device)
    date = datetime.now().strftime('%m%d_%H_%M')
    param_path = params["log_dir"] + '/ppo' + '/%s_%s_param.csv' % (date, "train")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))

    epoch = 0
    ave_act_loss = 0.0
    ave_cri_loss = 0.0

    act_model = PtrNet1(params).to(device)
    checkpoint = torch.load('experiment224/output/' + '{}_act.pt'.format(file_name))
    act_model.load_state_dict(checkpoint['model_state_dict_actor'])

    baseline_model = PtrNet1(params).to(device)  # baseline_model 불필요
    baseline_model.load_state_dict(act_model.state_dict())  # baseline_model 불필요
    if params["optimizer"] == 'Adam':
        latent_optim = optim.Adam(act_model.Latent.parameters(), lr=1.0e-4)
        act_optim = optim.Adam(act_model.all_attention_params, lr=params["lr"])
        cri_optim = optim.Adam(act_model.critic.parameters(), lr=params['lr'])
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"], gamma=params["lr_decay"])
        latent_lr_scheduler = optim.lr_scheduler.StepLR(latent_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])
        """
        act_model이라는 신경망 뭉치에 파라미터(가중치, 편향)을 업데이트 할꺼야.

        """

    elif params["optimizer"] == "RMSProp":
        act_optim = optim.RMSprop(act_model.parameters(), lr=params["lr"])
    # if params["is_lr_decay"]:
    #     act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
    #                                                  gamma=params["lr_decay"])

    t1 = time()
    ave_makespan = 0

    c_max = list()
    b = 0
    problem_list = [1, 2]
    validation_records_min = [[] for _ in problem_list]
    validation_records_mean = [[] for _ in problem_list]
    empty_records = [[], []]
    for s in range(epoch + 1, params["step"]):

        """

        변수별 shape 
        inputs : batch_size X number_of_blocks X number_of_process
        pred_seq : batch_size X number_of_blocks

        """
        b += 1

        if s % 100 == 1:  # Evaluation 수행



            for p in problem_list:
                min_makespan = heuristic_eval(p)
                eval_number = 30
                with torch.no_grad():
                    min_makespan_list = [min_makespan] * eval_number
                    min_makespan1, mean_makespan1 = evaluation(act_model, baseline_model, p, eval_number, device,
                                                               upperbound=min_makespan_list)


                min_makespan = min_makespan1
                mean_makespan = mean_makespan1
                if p == 1:
                    mean_makespan71 = mean_makespan1
                    min_makespan71 = min_makespan1
                else:
                    mean_makespan72 = mean_makespan1
                    min_makespan72 = min_makespan1

                print("TA{}".format(problem_list[p - 1]), min_makespan, mean_makespan)
                empty_records[p - 1].append(mean_makespan)
                #
                # if len(empty_records[1]) > 35 and np.mean(empty_records[1][-30:]) >= 3300:
                #     sys.exit()




if __name__ == '__main__':

    load_model = False

    log_dir = "./result/log"
    if not os.path.exists(log_dir + "/ppo"):
        os.makedirs(log_dir + "/ppo")

    model_dir = "./result/model"
    if not os.path.exists(model_dir + "/ppo_w_third_feature"):
        os.makedirs(model_dir + "/ppo_w_third_feature")

    params = {
        "num_of_process": 6,
        "step": cfg.step,
        "log_step": cfg.log_step,
        "log_dir": log_dir,
        "save_step": cfg.save_step,
        "model_dir": model_dir,
        "batch_size": cfg.batch_size,
        "init_min": -0.08,
        "init_max": 0.08,
        "use_logit_clipping": True,
        "C": cfg.C,
        "T": cfg.T,
        "iteration": cfg.iteration,
        "epsilon": float(os.environ.get("epsilon", 0.2)),
        "optimizer": "Adam",
        "n_glimpse": cfg.n_glimpse,
        "n_process": cfg.n_process,
        "lr_decay_step_critic": cfg.lr_decay_step_critic,
        "load_model": load_model,
        "entropy_weight": cfg.entropy_weight,
        "dot_product": cfg.dot_product,
        "lr_critic": cfg.lr_critic,

        "reward_scaler": cfg.reward_scaler,
        "beta": float(os.environ.get("beta", 0.65)),
        "alpha": float(os.environ.get("alpha", 0.1)),
        "lr": float(os.environ.get("lr", 5.0e-4)),
        "lr_decay": float(os.environ.get("lr_decay", 0.95)),
        "lr_decay_step": int(os.environ.get("lr_decay_step",500)),
        "layers": eval(str(os.environ.get("layers", '[256, 128]'))),
        "n_embedding": int(os.environ.get("n_embedding", 48)),
        "n_hidden": int(os.environ.get("n_hidden", 84)),
        "graph_embedding_size": int(os.environ.get("graph_embedding_size", 96)),
        "n_multi_head": int(os.environ.get("n_multi_head", 2)),
        "ex_embedding_size": int(os.environ.get("ex_embedding_size",36)),
        "k_hop": int(os.environ.get("k_hop", 1)),
        "is_lr_decay": True,
        "third_feature": 'first_and_second',  # first_and_second, first_only, second_only
        "baseline_reset": True,
        "ex_embedding": True,
        "w_representation_learning": False,
        "z_dim": 128,
        "k_epoch": int(os.environ.get("k_epoch", 2)),

    }

