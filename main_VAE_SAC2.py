import pandas as pd
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from time import time
from datetime import datetime
from actor_VAE3 import PtrNet1
from jssp2 import AdaptiveScheduler
from copy import deepcopy
from cfg import get_cfg
import random
from utils import *

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


def generate_jssp_instance(num_jobs, num_machine):
    temp = []
    for job in range(num_jobs):
        machine_sequence = list(range(num_machine))
        np.random.shuffle(machine_sequence)
        empty = list()
        for ops in range(num_machine):
            empty.append((machine_sequence[ops], np.random.randint(1, 100)))
        temp.append(empty)
    jobs_datas = []
    jobs_datas.append(temp)
    scheduler_list = list()
    scheduler_list.append(AdaptiveScheduler(jobs_datas[0]))
    return jobs_datas, scheduler_list


def evaluation(act_model, baseline_model, p, eval_number, device):
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
                                                   train=False)
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
    job_range = [5, 10]
    machine_range = [5, 10]
    act_model = PtrNet1(params).to(device)
    buffer = Replay_Buffer(buffer_size=20000, batch_size=params['batch_size'], job_range=job_range,
                           machine_range=machine_range)

    baseline_model = PtrNet1(params).to(device)  # baseline_model 불필요
    baseline_model.load_state_dict(act_model.state_dict())  # baseline_model 불필요
    if params["optimizer"] == 'Adam':
        latent_optim = optim.Adam(act_model.Latent.parameters(), lr=5e-5)
        act_optim = optim.Adam(act_model.all_attention_params, lr=3e-4)
        cri_optim = optim.Adam(act_model.critic.parameters(), lr=3e-4)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim.Adam([log_alpha], lr=5e-4)

        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])
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
                eval_number = 2
                with torch.no_grad():
                    min_makespan1, mean_makespan1 = evaluation(act_model, baseline_model, p, eval_number, device)

                min_makespan = min_makespan1
                mean_makespan = mean_makespan1

                print("TA{}".format(problem_list[p - 1]), min_makespan, mean_makespan)
                empty_records[p - 1].append(mean_makespan)
                #
                # if len(empty_records[1]) > 35 and np.mean(empty_records[1][-30:]) >= 3300:
                #     sys.exit()

                if cfg.vessl == True:
                    vessl.log(step=s, payload={'minmakespan{}'.format(str(problem_list[p - 1])): min_makespan})
                    vessl.log(step=s, payload={'meanmakespan{}'.format(str(problem_list[p - 1])): mean_makespan})
                else:
                    validation_records_min[p - 1].append(min_makespan)
                    validation_records_mean[p - 1].append(mean_makespan)
                    min_m = pd.DataFrame(validation_records_min)
                    mean_m = pd.DataFrame(validation_records_mean)
                    min_m = min_m.transpose()
                    mean_m = mean_m.transpose()
                    min_m.columns = problem_list
                    mean_m.columns = problem_list
                    min_m.to_csv('min_makespan_w_third_feature333.csv')
                    mean_m.to_csv('mean_makespan_w_third_feature333.csv')

        act_model.block_indices = []
        baseline_model.block_indices = []

        """
        훈련용 데이터셋 생성하는 코드
        """
        num_machine = np.random.randint(job_range[0], job_range[1])
        num_job = np.random.randint(num_machine, machine_range[1])
        jobs_datas, scheduler_list = generate_jssp_instance(num_jobs=num_job,
                                                            num_machine=num_machine)
        act_model.Latent.current_num_edges = num_machine * num_job
        for scheduler in scheduler_list:
            scheduler.reset()

        """

        """
        act_model.get_jssp_instance(scheduler_list)  # 훈련해야할 instance를 에이전트가 참조(등록)하는 코드
        act_model.eval()
        node_features = list()
        scheduler = AdaptiveScheduler(jobs_datas[0])
        node_feature = scheduler.get_node_feature()
        node_features.append(node_feature)
        edge_precedence = scheduler.get_edge_index_precedence()
        edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
        edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
        heterogeneous_edges = list()
        heterogeneous_edge = (edge_precedence, edge_antiprecedence, edge_machine_sharing)  # 세종류의 엑지들을 하나의 변수로 참조시킴
        heterogeneous_edges.append(heterogeneous_edge)
        input_data = (node_features, heterogeneous_edges)
        act_model.eval()
        pred_seq, ll_old, _, _, _, _, _ = act_model(input_data,
                                                       device,
                                                       scheduler_list=scheduler_list,
                                                       num_machine=num_machine,
                                                       num_job=num_job,
                                                       train=False
                                                       )

        sequence = pred_seq[0]
        makespan = -scheduler.run(sequence.tolist())/params['reward_scaler']
        c_max.append(makespan)
        buffer.memory(jobs_datas[0], pred_seq[0].cpu().detach().numpy(), ll_old[0].cpu().detach().numpy().tolist(), makespan,num_job, num_machine)

        """
        soft actor critic

        """
        eligible_problem_sizes = \
            [
                ps for ps, count in buffer.step_count.items()
                if count > buffer.batch_size
            ]
        if len(eligible_problem_sizes) >= 1:
            sampled_jobs_datas, action_sequences, ll_olds, _, sampled_problem_size = buffer.sample()
            sampled_problem_size = sampled_problem_size.split('_')
            sampled_num_job = int(sampled_problem_size[0])
            sampled_num_machine = int(sampled_problem_size[1])
            # print(sampled_problem_size)

            sampled_scheduler_list = list()
            for n in range(len(sampled_jobs_datas)):
                sampled_scheduler_list.append(AdaptiveScheduler(sampled_jobs_datas[n]))

            act_model.get_jssp_instance(sampled_scheduler_list)  # 훈련해야할 instance를 에이전트가 참조(등록)하는 코드
            heterogeneous_edges = list()
            node_features = list()

            for n in range(params['batch_size']):
                """

                Instance를 명세하는 부분
                - Node feature: Operation의 다섯개 feature
                - Edge: 

                """

                scheduler = AdaptiveScheduler(sampled_jobs_datas[n])
                node_feature = scheduler.get_node_feature()
                node_features.append(node_feature)
                edge_precedence = scheduler.get_edge_index_precedence()
                edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
                edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
                heterogeneous_edge = (
                edge_precedence, edge_antiprecedence, edge_machine_sharing)  # 세종류의 엑지들을 하나의 변수로 참조시킴
                heterogeneous_edges.append(heterogeneous_edge)

            sampled_input_data = (node_features, heterogeneous_edges)
            copied_sampled_input_data = deepcopy(sampled_input_data)
            copied_sampled_scheduler_list = deepcopy(sampled_scheduler_list)
            act_model.train()
            act_model.Latent.current_num_edges = sampled_num_machine * sampled_num_job

            pred_seq, lls, _, edge_loss, node_loss, loss_kld, q = act_model(sampled_input_data,
                                                                                                    device,
                                                                                                    scheduler_list=sampled_scheduler_list,
                                                                                                    num_machine=sampled_num_machine,
                                                                                                    num_job=sampled_num_job,
                                                                                                    train=True,
                                                                                                    q_update = True
                                                                                                    )
            sampled_makespans = list()
            for n in range(params['batch_size']):
                scheduler = AdaptiveScheduler(sampled_jobs_datas[n])
                makespan = -scheduler.run(pred_seq.tolist()[n])
                sampled_makespans.append(makespan)

            """

            1. Loss 구하기
            2. Gradient 구하기 (loss.backward)
            3. Update 하기(act_optim.step)

            """
            latent_loss = edge_loss + node_loss + loss_kld
            cri_loss = F.mse_loss(q.squeeze(1), torch.tensor(sampled_makespans).float().to(device))



            total_loss = latent_loss + cri_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=float(os.environ.get("grad_clip", 10)),
                                     norm_type=2)


            latent_optim.step()
            cri_optim.step()
            cri_optim.zero_grad()
            latent_optim.zero_grad()

            #if cri_loss.detach().cpu().numpy().tolist() <= 300000:
            pred_seq, lls, _, edge_loss, node_loss, loss_kld, q = act_model(copied_sampled_input_data,
                                                                                                    device,
                                                                                                    scheduler_list=copied_sampled_scheduler_list,
                                                                                                    num_machine=sampled_num_machine,
                                                                                                    num_job=sampled_num_job,
                                                                                                    train=True
                                                                                                    )

            act_loss = -(q.squeeze(1)-lls).mean()
            act_loss.backward()

            print("critic loss : ", np.round(cri_loss.detach().cpu().numpy().tolist(), 2), " q : ", np.round(q.detach().mean().cpu().numpy().tolist(), 2), " act loss : ",np.round(act_loss.detach().cpu().numpy().tolist(), 2), " sample makespan : ", np.round(torch.tensor(sampled_makespans).float().to(device).mean().detach().cpu().numpy().tolist(), 2))
            if s %  50 == 0:
                param_to_name = {}
                for name, param in act_model.named_parameters():
                    param_to_name[param] = name
                print("============================================================")
                for i, param in enumerate(act_model.critic.parameters()):
                    param_name = param_to_name.get(param, f"unknown_param_{i}")

                    if param.grad is not None:
                        print(f"[{i}] {param_name}: 그래디언트 있음, norm: {param.grad.norm().item():.12f}")
                    else:
                        print(f"[{i}] {param_name}: 그래디언트 없음")
                print("---------------------------------")
                for i, param in enumerate(act_model.all_attention_params):
                    param_name = param_to_name.get(param, f"unknown_param_{i}")

                    if param.grad is not None:
                        print(f"[{i}] {param_name}: 그래디언트 있음, norm: {param.grad.norm().item():.12f}")
                    else:
                        print(f"[{i}] {param_name}: 그래디언트 없음")
            act_optim.step()
            act_optim.zero_grad()
            ave_act_loss += act_loss.item()

            if s % params["log_step"] == 0:
                t2 = time()
                print('step:{}/{}, actic loss:{:1.3f}, q value:{:1.3f}, entropy:{:1.3f}, {}min{}sec'.format(
                    s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]),
                                       q.squeeze(1).mean().cpu().detach().numpy().tolist(), lls.mean().cpu().detach().numpy().tolist(), (t2 - t1) // 60,
                                       (t2 - t1) % 60))

            if s % params["save_step"] == 1:
                if cfg.vessl == False:
                    torch.save({'epoch': s,
                                'model_state_dict_actor': act_model.state_dict(),
                                'optimizer_state_dict_actor': act_optim.state_dict(),
                                'ave_act_loss': ave_act_loss,
                                'ave_cri_loss': 0},
                               params["model_dir"] + '/ppo_w_third_feature' + '/%s_step%d_act.pt' % (date, s))
                else:
                    torch.save({'epoch': s,
                                'model_state_dict_actor': act_model.state_dict(),
                                'optimizer_state_dict_actor': act_optim.state_dict(),
                                'ave_act_loss': ave_act_loss,
                                'ave_cri_loss': 0},
                               output_dir + '/%s_step%d_act.pt' % (date, s))

            #     print('save model...')


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
        "batch_size": 12,
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
        "alpha": float(os.environ.get("alpha", 0.05)),
        "lr": float(os.environ.get("lr", 1.0e-3)),
        "lr_decay": float(os.environ.get("lr_decay", 0.95)),
        "lr_decay_step": int(os.environ.get("lr_decay_step", 500)),
        "layers": eval(str(os.environ.get("layers", '[196, 84]'))),
        "n_embedding": int(os.environ.get("n_embedding", 48)),
        "n_hidden": int(os.environ.get("n_hidden", 72)),
        "graph_embedding_size": int(os.environ.get("graph_embedding_size", 96)),
        "n_multi_head": int(os.environ.get("n_multi_head", 1)),
        "ex_embedding_size": int(os.environ.get("ex_embedding_size", 42)),

        "k_hop": int(os.environ.get("k_hop", 1)),
        "is_lr_decay": True,
        "third_feature": 'first_and_second',  # first_and_second, first_only, second_only
        "baseline_reset": True,
        "ex_embedding": True,
        "k_epoch": int(os.environ.get("k_epoch", 2)),
        "w_representation_learning": True,
    }

    train_model(params)