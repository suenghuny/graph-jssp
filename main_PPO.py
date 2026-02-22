import pandas as pd
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import wandb

import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from utils import *
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


def step_with_min(scheduler, optimizer, min_lr=1e-6):
    scheduler.step()
    for param_group in optimizer.param_groups:
        param_group['lr'] = max(param_group['lr'], min_lr)


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

if cfg.additional_validation == True:
    for i in ["ta21", "ta22", "dmu46", "dmu47", "ta61", "ta62", "dmu76", "dmu77"]:
        df = pd.read_excel("eval_dataset.xlsx", sheet_name=i, engine='openpyxl')
        orb_data = list()  #
        for row, column in df.iterrows():
            job = []
            for j in range(0, len(column.tolist()), 2):
                element = (column.tolist()[j], column.tolist()[j + 1])
                job.append(element)
            orb_data.append(job)
        orb_list.append(orb_data)

else:
    for i in ["ta61", "ta62", "dmu76", "dmu77"]:
        df = pd.read_excel("eval_dataset.xlsx", sheet_name=i, engine='openpyxl')
        orb_data = list()  #
        for row, column in df.iterrows():
            job = []
            for j in range(0, len(column.tolist()), 2):
                element = (column.tolist()[j], column.tolist()[j + 1])
                job.append(element)
            orb_data.append(job)
        orb_list.append(orb_data)


def generate_jssp_instance(num_jobs, num_machine, batch_size):
    jobs_datas = []
    for _ in range(batch_size):
        temp = []
        for job in range(num_jobs):
            machine_sequence = list(range(num_machine))
            np.random.shuffle(machine_sequence)
            empty = list()
            for ops in range(num_machine):
                empty.append((machine_sequence[ops], np.random.randint(1, 100)))
            temp.append(empty)
        jobs_datas.append(temp)
    scheduler_list = list()
    for n in range(len(jobs_datas)):
        scheduler_list.append(AdaptiveScheduler(jobs_datas[n]))
    return jobs_datas, scheduler_list


def heuristic_eval(p):  # 불필요(현재로써는)
    scheduler = AdaptiveScheduler(orb_list[p - 1])
    makespan_heu = scheduler.heuristic_run()
    return makespan_heu


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


# 0628_20_59_step86000_act_w_rep

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
                                                train=False)
    for sequence in pred_seq:
        scheduler = AdaptiveScheduler(orb_list[p - 1])
        makespan = scheduler.run(sequence.tolist())
        val_makespan.append(makespan)
    # print("크크크", val_makespan)
    return np.min(val_makespan), np.mean(val_makespan)


def train_model(params, selected_param, log_path=None):
    wandb.login()
    s_latent = params["s_latent"]
    aggr = params['aggr']

    if cfg.algo == 'reinforce':
        if params["w_representation_learning"] == True:
            wandb.init(project="Graph JSSP", name=selected_param + 'w_rep_{}'.format(aggr))
        else:
            wandb.init(project="Graph JSSP", name=selected_param + 'wo_rep_{}'.format(aggr))
    elif cfg.algo == 'rep_learning':
        wandb.init(project="Graph JSSP", name=selected_param + 'seperation_s_latent_{}_{}'.format(s_latent, aggr))
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    date = datetime.now().strftime('%m%d_%H_%M')
    param_path = params["log_dir"] + '/ppo' + '/%s_%s_param.csv' % (date, "train")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))

    epoch = 0
    ave_act_loss = 0.0
    ave_cri_loss = 0.0

    act_model = PtrNet1(params).to(device)

    baseline_model = PtrNet1(params).to(device)  # baseline_model 불필요
    baseline_model.load_state_dict(act_model.state_dict())  # baseline_model 불필요
    if params["w_representation_learning"] == True:
        latent_optim = optim.Adam(act_model.Latent.parameters(), lr=params["lr_latent"])
        act_optim = optim.Adam(list(act_model.all_attention_params) + list(act_model.critic.parameters()),
                               lr=params["lr"])
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])
        latent_lr_scheduler = optim.lr_scheduler.StepLR(latent_optim, step_size=params["lr_decay_step"],
                                                        gamma=params["lr_decay"])
        """
        act_model이라는 신경망 뭉치에 파라미터(가중치, 편향)을 업데이트 할꺼야.

        """

    else:
        act_optim = optim.Adam(act_model.parameters(), lr=params["lr"])
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])

    # if params["is_lr_decay"]:
    #     act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
    #                                                  gamma=params["lr_decay"])

    t1 = time()
    ave_makespan = 0

    c_max = list()
    b = 0
    if cfg.additional_validation == True:
        problem_list = [1, 2, 3, 4, 5, 6, 7, 8]
    else:
        problem_list = [1, 2, 3, 4]
    validation_records_min = [[] for _ in problem_list]
    validation_records_mean = [[] for _ in problem_list]
    empty_records = [[] for _ in problem_list]

    for s in range(epoch + 1, params["step"]):

        """

        변수별 shape 
        inputs : batch_size X number_of_blocks X number_of_process
        pred_seq : batch_size X number_of_blocks

        """
        b += 1

        if s % 200 == 1:  # Evaluation 수행
            if cfg.algo == 'reinforce':
                for p in problem_list:

                    eval_number = 1
                    with torch.no_grad():
                        min_makespan1, mean_makespan1 = evaluation(act_model, baseline_model, p, eval_number, device)
                    min_makespan = min_makespan1
                    mean_makespan = mean_makespan1

                    if cfg.additional_validation == True:
                        if p == 1:
                            mean_makespan21 = mean_makespan1
                            print("TA21", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        elif p == 2:
                            mean_makespan22 = mean_makespan1
                            print("TA22", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        elif p == 3:
                            mean_makespan46 = mean_makespan1
                            print("DMU46", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        elif p == 4:
                            mean_makespan47 = mean_makespan1
                            print("DMU47", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        elif p == 5:
                            mean_makespan61 = mean_makespan1
                            print("TA61", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        elif p == 6:
                            mean_makespan62 = mean_makespan1
                            print("TA62", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        elif p == 7:
                            mean_makespan76 = mean_makespan1
                            print("DMU76", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        else:
                            mean_makespan77 = mean_makespan1
                            print("DMU77", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)

                        if params['w_representation_learning'] == True:
                            min_m.to_csv('additional_val_w_rep_min_makespan_{}.csv'.format(selected_param))
                            mean_m.to_csv('additional_val_w_rep_mean_makespan_{}.csv'.format(selected_param))



                    else:
                        if p == 1:
                            mean_makespan61 = mean_makespan1
                            min_makespan61 = min_makespan1
                            print("TA61", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        elif p == 2:
                            mean_makespan62 = mean_makespan1
                            min_makespan62 = min_makespan1
                            print("TA62", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        elif p == 3:
                            mean_makespan76 = mean_makespan1
                            min_makespan76 = min_makespan1
                            print("DMU76", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        else:
                            mean_makespan77 = mean_makespan1
                            min_makespan77 = min_makespan1
                            print("DMU77", min_makespan, mean_makespan)
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

                            t1 = time()

                            if params['w_representation_learning'] == True:
                                if cfg.continuous_bernoulli == False:
                                    min_m.to_csv('w_rep_min_makespan_{}.csv'.format(selected_param))
                                    mean_m.to_csv('w_rep_mean_makespan_{}.csv'.format(selected_param))
                                else:
                                    min_m.to_csv('cont_bernoulli_w_rep_min_makespan_{}.csv'.format(selected_param))
                                    mean_m.to_csv('cont_bernoulli_w_rep_mean_makespan_{}.csv'.format(selected_param))
                            else:
                                min_m.to_csv('wo_rep_min_makespan_{}.csv'.format(selected_param))
                                mean_m.to_csv('wo_rep_mean_makespan_{}.csv'.format(selected_param))

                wandb.log({
                    "episode": s,
                    "TA61 makespan": mean_makespan61,
                    "TA62 makespan": mean_makespan62,
                    "DMU76 makespan": mean_makespan76,
                    "DMU77 makespan": mean_makespan77,
                })
                print(s, "save")
                if params['w_representation_learning'] == True:
                    torch.save({'epoch': s,
                                'model_state_dict_actor': act_model.state_dict(),
                                'optimizer_state_dict_actor': act_optim.state_dict(),
                                'ave_act_loss': ave_act_loss,
                                'ave_cri_loss': 0,
                                'ave_makespan': ave_makespan},
                               params["model_dir"] + '/w_rep_{}_step_{}_mean_makespan_{}.pt'.format(selected_param, s,
                                                                                                    mean_makespan61))
                else:
                    torch.save({'epoch': s,
                                'model_state_dict_actor': act_model.state_dict(),
                                'optimizer_state_dict_actor': act_optim.state_dict(),
                                'ave_act_loss': ave_act_loss,
                                'ave_cri_loss': 0,
                                'ave_makespan': ave_makespan},
                               params["model_dir"] + '/wo_rep_{}_step_{}_mean_makespan_{}.pt'.format(selected_param, s,
                                                                                                     mean_makespan61))
            elif cfg.algo == 'rep_learning':
                if s <= s_latent:
                    pass
                else:
                    for p in problem_list:

                        eval_number = 1
                        with torch.no_grad():
                            min_makespan1, mean_makespan1 = evaluation(act_model, baseline_model, p, eval_number,
                                                                       device)
                        min_makespan = min_makespan1
                        mean_makespan = mean_makespan1
                        if p == 1:
                            mean_makespan61 = mean_makespan1
                            min_makespan61 = min_makespan1
                            print("TA61", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        elif p == 2:
                            mean_makespan62 = mean_makespan1
                            min_makespan62 = min_makespan1
                            print("TA62", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        elif p == 3:
                            mean_makespan76 = mean_makespan1
                            min_makespan76 = min_makespan1
                            print("DMU76", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)
                        else:
                            mean_makespan77 = mean_makespan1
                            min_makespan77 = min_makespan1
                            print("DMU77", min_makespan, mean_makespan)
                            empty_records[p - 1].append(mean_makespan)

                        #
                        # if len(empty_records[1]) > 35 and np.mean(empty_records[1][-30:]) >= 3300:
                        #     sys.exit()

                        if cfg.vessl == True:
                            vessl.log(step=s, payload={'minmakespan{}'.format(str(problem_list[p - 1])): min_makespan})
                            vessl.log(step=s,
                                      payload={'meanmakespan{}'.format(str(problem_list[p - 1])): mean_makespan})
                        else:
                            validation_records_min[p - 1].append(min_makespan)
                            validation_records_mean[p - 1].append(mean_makespan)
                            min_m = pd.DataFrame(validation_records_min)
                            mean_m = pd.DataFrame(validation_records_mean)
                            min_m = min_m.transpose()
                            mean_m = mean_m.transpose()
                            min_m.columns = problem_list
                            mean_m.columns = problem_list

                            t1 = time()
                            if cfg.feature_selection_mode==True:
                                mean_m.to_csv(
                                    'feature_selection_mode_{}_seperation_after_rep_{}_{}_mean_makespan_{}.csv'.format(
                                   cfg.exclude_feature,
                                   s_latent, selected_param,
                                   mean_makespan61))
                            else:
                                if cfg.loss_type == 'cross_entropy':
                                    mean_m.to_csv('seperation_after_rep_{}_{}_mean_makespan_{}.csv'.format(
                                       s_latent, selected_param,
                                       mean_makespan61))
                                elif cfg.loss_type == 'continuous_bernoulli':
                                    mean_m.to_csv('continuous_bernoulli_seperation_after_rep_{}_{}_mean_makespan_{}.csv'.format(
                                        s_latent, selected_param,
                                        mean_makespan61))
                                elif cfg.loss_type == 'mse':
                                    mean_m.to_csv(
                                        'mse_seperation_after_rep_{}_{}_mean_makespan_{}.csv'.format(
                                            s_latent, selected_param,
                                            mean_makespan61))

                    wandb.log({
                        "episode": s,
                        "TA61 makespan": mean_makespan61,
                        "TA62 makespan": mean_makespan62,
                        "DMU76 makespan": mean_makespan76,
                        "DMU77 makespan": mean_makespan77,
                    })
                    print(s, "save")
                    torch.save({'epoch': s,
                                'model_state_dict_actor': act_model.state_dict(),
                                'optimizer_state_dict_actor': act_optim.state_dict(),
                                'ave_act_loss': ave_act_loss,
                                'ave_cri_loss': 0,
                                'ave_makespan': ave_makespan},

                               params["model_dir"] + '/seperation_after_rep_{}_{}_step_{}_mean_makespan_{}.pt'.format(
                                   s_latent, selected_param, s,
                                   mean_makespan61))

        act_model.block_indices = []
        baseline_model.block_indices = []

        if s % cfg.gen_step == 1:  # 훈련용 Data Instance 새롭게 생성 (gen_step 마다 생성)
            """
            훈련용 데이터셋 생성하는 코드
            """
            num_machine = np.random.randint(5, 10)
            num_job = np.random.randint(num_machine, 10)

            jobs_datas, scheduler_list = generate_jssp_instance(num_jobs=num_job,
                                                                num_machine=num_machine,
                                                                batch_size=params['batch_size'])
            act_model.Latent.current_num_edges = num_machine * num_job
            # act_model.Latent.decoder.max_nodes = num_machine * num_job
            makespan_list_for_upperbound = list()
            for scheduler in scheduler_list:
                c_max_heu = scheduler.heuristic_run()
                makespan_list_for_upperbound.append(c_max_heu)
                scheduler.reset()
        else:
            for scheduler in scheduler_list:
                scheduler.reset()
        """

        """
        act_model.get_jssp_instance(scheduler_list)  # 훈련해야할 instance를 에이전트가 참조(등록)하는 코드
        heterogeneous_edges = list()
        node_features = list()

        for n in range(params['batch_size']):
            """
            Instance를 명세하는 부분
            - Node feature: Operation의 다섯개 feature
            - Edge: 

            """

            scheduler = AdaptiveScheduler(jobs_datas[n])

            node_feature = scheduler.get_node_feature()
            node_features.append(node_feature)
            edge_precedence = scheduler.get_edge_index_precedence()
            edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
            edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
            heterogeneous_edge = (edge_precedence, edge_antiprecedence, edge_machine_sharing)  # 세종류의 엑지들을 하나의 변수로 참조시킴
            heterogeneous_edges.append(heterogeneous_edge)
        input_data = (node_features, heterogeneous_edges)
        act_model.train()
        if cfg.algo == 'reinforce':

            pred_seq, ll_old, _, edge_loss, node_loss, loss_kld, baselines = act_model(input_data,
                                                                                       device,
                                                                                       scheduler_list=scheduler_list,
                                                                                       num_machine=num_machine,
                                                                                       num_job=num_job
                                                                                       )
            real_makespan = list()
            for n in range(pred_seq.shape[0]):  # act_model(agent)가 산출한 해를 평가하는 부분
                sequence = pred_seq[n]
                scheduler = AdaptiveScheduler(jobs_datas[n])
                makespan = -scheduler.run(sequence.tolist()) / params['reward_scaler']
                real_makespan.append(makespan)
                c_max.append(makespan)
            ave_makespan += sum(real_makespan) / (params["batch_size"] * params["log_step"])
            """
            vanila actor critic
            """

            entropy = -ll_old  # entropy = -E[log(p)]
            initial_coeff = params['entropy_coeff']
            target_coeff = params['entropy_min']
            anneal_step = params['rep_anneal']
            entrop_coeff = max(target_coeff, initial_coeff - (initial_coeff - target_coeff) * s / anneal_step)
            entropy_loss = entrop_coeff * entropy

            # target_entropy = params["target_entropy"]
            # log_alpha_loss = -act_model.log_alpha * (ll_old.detach() + target_entropy).mean()
            adv = torch.tensor(real_makespan).detach().unsqueeze(1).to(device) - baselines  # baseline(advantage) 구하는 부분
            cri_loss = F.mse_loss(torch.tensor(real_makespan).to(device) + entropy_loss.detach().squeeze(1),
                                  baselines.squeeze(1))

            """

            1. Loss 구하기
            2. Gradient 구하기 (loss.backward)
            3. Update 하기(act_optim.step)

            """
            latent_loss = edge_loss + node_loss + loss_kld
            act_loss = -(ll_old * adv.detach() + entropy_loss).mean()  # loss 구하는 부분 /  ll_old의 의미 log_theta (pi | s)

            if params['w_representation_learning'] == True:
                total_loss = latent_loss + act_loss + cri_loss  # +log_alpha_loss
                latent_optim.zero_grad()
                act_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=float(os.environ.get("grad_clip", 5)),
                                         norm_type=2)
                latent_optim.step()
                act_optim.step()
                step_with_min(act_lr_scheduler, act_optim, min_lr=params['lr_decay_min'])
                step_with_min(latent_lr_scheduler, latent_optim, min_lr=1e-5)
            else:
                total_loss = act_loss + cri_loss
                act_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=float(os.environ.get("grad_clip", 5)),
                                         norm_type=2)
                act_optim.step()
                step_with_min(act_lr_scheduler, act_optim, min_lr=params['lr_decay_min'])
            ave_act_loss += act_loss.item()
            if s % params["log_step"] == 0:
                t2 = time()
                if params['w_representation_learning'] == True:
                    print(
                        'with representation learning step:%d/%d, actic loss:%1.3f, crictic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                            s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]),
                            ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan, (t2 - t1) // 60,
                            (t2 - t1) % 60))
                else:
                    print(
                        'without representation learning step:%d/%d, actic loss:%1.3f, crictic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                            s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]),
                            ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan, (t2 - t1) // 60,
                            (t2 - t1) % 60))
        elif cfg.algo == 'rep_learning':
            if s <= s_latent:
                edge_loss, node_loss, loss_kld = act_model.forward_latent(input_data,
                                                                          device,
                                                                          scheduler_list=scheduler_list,
                                                                          num_machine=num_machine,
                                                                          num_job=num_job
                                                                          )

                latent_loss = edge_loss + node_loss + loss_kld
                total_loss = latent_loss
                latent_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=float(os.environ.get("grad_clip", 5)),
                                         norm_type=2)
                latent_optim.step()
                step_with_min(latent_lr_scheduler, latent_optim, min_lr=5e-5)
                print("representation_learning {}".format(s))
            else:
                pred_seq, ll_old, _, edge_loss, node_loss, loss_kld, baselines = act_model(input_data,
                                                                                           device,
                                                                                           scheduler_list=scheduler_list,
                                                                                           num_machine=num_machine,
                                                                                           num_job=num_job
                                                                                           )
                real_makespan = list()
                for n in range(pred_seq.shape[0]):  # act_model(agent)가 산출한 해를 평가하는 부분
                    sequence = pred_seq[n]
                    scheduler = AdaptiveScheduler(jobs_datas[n])
                    makespan = -scheduler.run(sequence.tolist()) / params['reward_scaler']
                    real_makespan.append(makespan)
                    c_max.append(makespan)
                ave_makespan += sum(real_makespan) / (params["batch_size"] * params["log_step"])
                """
                vanila actor critic
                """

                entropy = -ll_old  # entropy = -E[log(p)]
                initial_coeff = params['entropy_coeff']

                target_coeff = params['entropy_min']
                anneal_step = params['rep_anneal']
                entrop_coeff = max(target_coeff,
                                   initial_coeff - (initial_coeff - target_coeff) * (s - s_latent) / anneal_step)
                entropy_loss = entrop_coeff * entropy

                # target_entropy = params["target_entropy"]
                # log_alpha_loss = -act_model.log_alpha * (ll_old.detach() + target_entropy).mean()
                adv = torch.tensor(real_makespan).detach().unsqueeze(1).to(
                    device) - baselines  # baseline(advantage) 구하는 부분
                cri_loss = F.mse_loss(torch.tensor(real_makespan).to(device) + entropy_loss.detach().squeeze(1),
                                      baselines.squeeze(1))

                """

                1. Loss 구하기
                2. Gradient 구하기 (loss.backward)
                3. Update 하기(act_optim.step)

                """
                act_loss = -(
                            ll_old * adv.detach() + entropy_loss).mean()  # loss 구하는 부분 /  ll_old의 의미 log_theta (pi | s)

                total_loss = act_loss + cri_loss  # +log_alpha_loss
                act_optim.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(act_model.parameters(),
                                         max_norm=float(os.environ.get("grad_clip", 5)),
                                         norm_type=2)
                act_optim.step()
                step_with_min(act_lr_scheduler, act_optim, min_lr=params['lr_decay_min'])

                ave_act_loss += act_loss.item()
                if s % params["log_step"] == 0:
                    t2 = time()
                    if params['w_representation_learning'] == True:
                        print(
                            'with representation learning step:%d/%d, actic loss:%1.3f, crictic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                                s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]),
                                ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan, (t2 - t1) // 60,
                                (t2 - t1) % 60))
                    else:
                        print(
                            'without representation learning step:%d/%d, actic loss:%1.3f, crictic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                                s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]),
                                ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan, (t2 - t1) // 60,
                                (t2 - t1) % 60))


def test(act_model, baseline_model, p, eval_number, device, problems, upperbound=None):
    scheduler_list_val = [AdaptiveScheduler(problems[p - 1]) for _ in range(eval_number)]
    val_makespan = list()
    act_model.get_jssp_instance(scheduler_list_val)
    baseline_model.get_jssp_instance(scheduler_list_val)

    act_model.eval()

    scheduler = AdaptiveScheduler(problems[p - 1])  # scheduler는 validation(ORB set)에 대해 수행

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
        scheduler = AdaptiveScheduler(problems[p - 1])
        makespan = scheduler.run(sequence.tolist())
        val_makespan.append(makespan)
    # print("크크크", val_makespan)
    return np.min(val_makespan), np.mean(val_makespan)


def test_model(params, selected_param, log_path=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    date = datetime.now().strftime('%m%d_%H_%M')
    param_path = params["log_dir"] + '/ppo' + '/%s_%s_param.csv' % (date, "train")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))
    act_model = PtrNet1(params).to(device)
    baseline_model = PtrNet1(params).to(device)  # baseline_model 불필요
    baseline_model.load_state_dict(act_model.state_dict())  # baseline_model 불필요
    file_name = 'seperation_after_rep_80000_param4_step_86001_mean_makespan_3181.0'  ###########fdfdf
    checkpoint = torch.load('{}.pt'.format(file_name))
    act_model.load_state_dict(checkpoint['model_state_dict_actor'])

    for dataset_name in ['ta', 'dmu', 'orb', 'la', 'abz', 'swv', 'ft', 'yn']:
        excel_file = pd.ExcelFile("{}_structured.xlsx".format(dataset_name), engine='openpyxl')
        total_sheets = len(excel_file.sheet_names)
        orb_list = []
        problem_number = [str(i + 1) for i in range(total_sheets)]
        print(problem_number)
        for i in problem_number:
            df = pd.read_excel("{}_structured.xlsx".format(dataset_name), sheet_name=i, engine='openpyxl')
            orb_data = list()  #
            for row, column in df.iterrows():
                job = []
                for j in range(0, len(column.tolist()), 2):
                    element = (column.tolist()[j], column.tolist()[j + 1])
                    job.append(element)
                orb_data.append(job)
            orb_list.append(orb_data)
        problem_list = [int(p) for p in problem_number]
        empty_records = dict()


        for p in problem_list:
            eval_number = 1
            with torch.no_grad():
                min_makespan1, mean_makespan1 = test(act_model, baseline_model, p, eval_number, device,  problems = orb_list )
            print("{}{} : {}".format(dataset_name,p, min_makespan1))
            empty_records[str(p)] = [min_makespan1]
            #print(empty_records)
            test_m = pd.DataFrame(empty_records).transpose()
            test_m.to_csv('{}_test_{}.csv'.format(dataset_name, file_name))


#
# def visualize_model(params, selected_param, log_path=None):
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     date = datetime.now().strftime('%m%d_%H_%M')
#     param_path = params["log_dir"] + '/ppo' + '/%s_%s_param.csv' % (date, "train")
#     print(f'generate {param_path}')
#     with open(param_path, 'w') as f:
#         f.write(''.join('%s,%s\n' % item for item in params.items()))
#
#     # 모델 로딩 부분 (기존과 동일)
#     act_model = PtrNet1(params).to(device)
#     baseline_model = PtrNet1(params).to(device)
#     baseline_model.load_state_dict(act_model.state_dict())
#     file_name = 'after_rep_40000_param0_step_90901_mean_makespan_3328.0'
#     checkpoint = torch.load('result/model/' + '{}.pt'.format(file_name))
#     act_model.load_state_dict(checkpoint['model_state_dict_actor'])
#
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from sklearn.decomposition import PCA
#     from sklearn.manifold import TSNE
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.cluster import KMeans
#     from sklearn.mixture import GaussianMixture
#     from sklearn.metrics import silhouette_score
#     import seaborn as sns
#     from scipy.stats import multivariate_normal
#     import umap
#
#     # 데이터 수집
#     flow_shop_index_list = list()
#     bottleneck_index_list = list()
#     z_list = list()
#
#     for i in range(2000):
#         num_machine = np.random.randint(5, 10)
#         num_job = np.random.randint(num_machine, 10)
#         jobs_datas, scheduler_list = generate_jssp_instance(num_jobs=num_job, num_machine=num_machine, batch_size=1)
#         act_model.Latent.current_num_edges = num_machine * num_job
#         flow_shop_index = calculate_flow_shop_index(scheduler_list[0])
#         bottleneck_index = calculate_bottleneck_index(scheduler_list[0])
#
#         scheduler = AdaptiveScheduler(jobs_datas[0])
#         node_features = [scheduler.get_node_feature()]
#         edge_precedence = scheduler.get_edge_index_precedence()
#         edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
#         edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
#         heterogeneous_edges = [(edge_precedence, edge_antiprecedence, edge_machine_sharing)]
#         input_data = (node_features, heterogeneous_edges)
#         z = act_model.forward_latent(input_data, device, scheduler_list=scheduler_list,
#                                      num_machine=num_machine, num_job=num_job, visualize=True)
#
#         flow_shop_index_list.append(flow_shop_index)
#         bottleneck_index_list.append(bottleneck_index)
#         z_list.append(z.squeeze(0).detach().cpu().numpy())
#         print("data collecting : ", i)
#
#     z_array = np.array(z_list)
#     flow_shop_index_list = np.array(flow_shop_index_list)
#     bottleneck_index_list = np.array(bottleneck_index_list)
#
#     print(f"z_array shape: {z_array.shape}")
#
#     # 1. PCA - 선형 변환으로 주성분 찾기 (Gaussian에 적합)
#     print("PCA 분석 중...")
#     pca = PCA(n_components=min(10, z_array.shape[1]))
#     z_pca = pca.fit_transform(z_array)
#
#     # 2. Gaussian Mixture Model로 클러스터링
#     print("Gaussian Mixture Model 클러스터링 중...")
#     n_components_range = range(2, 8)
#     bic_scores = []
#     aic_scores = []
#
#     for n_comp in n_components_range:
#         gmm = GaussianMixture(n_components=n_comp, random_state=42)
#         gmm.fit(z_array)
#         bic_scores.append(gmm.bic(z_array))
#         aic_scores.append(gmm.aic(z_array))
#
#     # 최적 클러스터 수 선택 (BIC 기준)
#     optimal_n_components = n_components_range[np.argmin(bic_scores)]
#     gmm_optimal = GaussianMixture(n_components=optimal_n_components, random_state=42)
#     cluster_labels = gmm_optimal.fit_predict(z_array)
#
#     # 3. 시각화
#     fig = plt.figure(figsize=(20, 15))
#
#     # 3-1. PCA 결과 (처음 2개 주성분)
#     ax1 = plt.subplot(3, 4, 1)
#     scatter1 = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=flow_shop_index_list,
#                            cmap='RdBu', alpha=1.0, s=30)
#     plt.title('PCA: Flow Shop Index', fontweight='bold')
#     plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
#     plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
#     plt.colorbar(scatter1, label='Flow Shop Index')
#     plt.grid(True, alpha=0.3)
#
#     ax2 = plt.subplot(3, 4, 2)
#     scatter2 = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=bottleneck_index_list,
#                            cmap='RdBu', alpha=1.0, s=30)
#     plt.title('PCA: Bottleneck Index', fontweight='bold')
#     plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
#     plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
#     plt.colorbar(scatter2, label='Bottleneck Index')
#     plt.grid(True, alpha=0.3)
#
#     # 3-2. GMM 클러스터링 결과
#     ax3 = plt.subplot(3, 4, 3)
#     scatter3 = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=cluster_labels,
#                            cmap='tab10', alpha=1.0, s=30)
#     plt.title(f'GMM Clustering (k={optimal_n_components})', fontweight='bold')
#     plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
#     plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
#     plt.colorbar(scatter3, label='Cluster')
#     plt.grid(True, alpha=0.3)
#
#     # 3-3. PCA Explained Variance
#     ax4 = plt.subplot(3, 4, 4)
#     cumsum_var = np.cumsum(pca.explained_variance_ratio_)
#     plt.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'bo-')
#     plt.title('PCA Explained Variance', fontweight='bold')
#     plt.xlabel('Principal Component')
#     plt.ylabel('Cumulative Explained Variance')
#     plt.grid(True, alpha=0.3)
#
#     # 3-4. 클러스터별 Flow Shop Index 분포
#     ax5 = plt.subplot(3, 4, 5)
#     for cluster_id in np.unique(cluster_labels):
#         mask = cluster_labels == cluster_id
#         plt.hist(flow_shop_index_list[mask], alpha=0.6, label=f'Cluster {cluster_id}', bins=15)
#     plt.title('Flow Shop Index by Cluster', fontweight='bold')
#     plt.xlabel('Flow Shop Index')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#
#     # 3-5. 클러스터별 Bottleneck Index 분포
#     ax6 = plt.subplot(3, 4, 6)
#     for cluster_id in np.unique(cluster_labels):
#         mask = cluster_labels == cluster_id
#         plt.hist(bottleneck_index_list[mask], alpha=0.6, label=f'Cluster {cluster_id}', bins=15)
#     plt.title('Bottleneck Index by Cluster', fontweight='bold')
#     plt.xlabel('Bottleneck Index')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#
#     # 3-6. BIC/AIC 스코어
#     ax7 = plt.subplot(3, 4, 7)
#     plt.plot(n_components_range, bic_scores, 'ro-', label='BIC')
#     plt.plot(n_components_range, aic_scores, 'bo-', label='AIC')
#     plt.title('Model Selection (GMM)', fontweight='bold')
#     plt.xlabel('Number of Components')
#     plt.ylabel('Information Criterion')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#
#     # 3-7. 잠재공간의 차원별 분산
#     ax8 = plt.subplot(3, 4, 8)
#     z_var = np.var(z_array, axis=0)
#     plt.plot(z_var, 'go-')
#     plt.title('Variance per Latent Dimension', fontweight='bold')
#     plt.xlabel('Latent Dimension')
#     plt.ylabel('Variance')
#     plt.grid(True, alpha=0.3)
#
#     # 3-8. 상관관계 히트맵 (처음 몇 개 차원만)
#     ax9 = plt.subplot(3, 4, 9)
#     n_dims_to_show = min(8, z_array.shape[1])
#     corr_matrix = np.corrcoef(z_array[:, :n_dims_to_show].T)
#     im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
#     plt.title(f'Latent Dimensions Correlation\n(First {n_dims_to_show} dims)', fontweight='bold')
#     plt.colorbar(im)
#
#     # 3-9. t-SNE (비교용)
#     print("t-SNE 계산 중...")
#     tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#     z_tsne = tsne.fit_transform(z_array)
#
#     ax10 = plt.subplot(3, 4, 10)
#     scatter10 = plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=flow_shop_index_list,
#                             cmap='RdBu', alpha=1.0, s=30)
#     plt.title('t-SNE: Flow Shop Index', fontweight='bold')
#     plt.xlabel('t-SNE 1')
#     plt.ylabel('t-SNE 2')
#     plt.colorbar(scatter10, label='Flow Shop Index')
#     plt.grid(True, alpha=0.3)
#
#     # 3-10. UMAP (비교용)
#     print("UMAP 계산 중...")
#     umap_reducer = umap.UMAP(n_components=2, random_state=42)
#     z_umap = umap_reducer.fit_transform(z_array)
#
#     ax11 = plt.subplot(3, 4, 11)
#     scatter11 = plt.scatter(z_umap[:, 0], z_umap[:, 1], c=flow_shop_index_list,
#                             cmap='RdBu', alpha=1.0, s=30)
#     plt.title('UMAP: Flow Shop Index', fontweight='bold')
#     plt.xlabel('UMAP 1')
#     plt.ylabel('UMAP 2')
#     plt.colorbar(scatter11, label='Flow Shop Index')
#     plt.grid(True, alpha=0.3)
#
#     # 3-11. 클러스터 중심과 분산 타원
#     ax12 = plt.subplot(3, 4, 12)
#     colors = plt.cm.tab10(np.linspace(0, 1, optimal_n_components))
#
#     for i, (mean, covar, color) in enumerate(zip(gmm_optimal.means_, gmm_optimal.covariances_, colors)):
#         # PCA 공간에서의 평균과 공분산
#         mean_pca = pca.transform(mean.reshape(1, -1))[0]
#
#         # 2D 부분만 사용 (처음 2개 주성분)
#         if z_pca.shape[1] >= 2:
#             cluster_mask = cluster_labels == i
#             plt.scatter(z_pca[cluster_mask, 0], z_pca[cluster_mask, 1],
#                         c=[color], alpha=0.6, s=30, label=f'Cluster {i}')
#             plt.scatter(mean_pca[0], mean_pca[1], c='black', s=200, marker='x')
#
#     plt.title('GMM Clusters with Centers', fontweight='bold')
#     plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
#     plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()
#
#     # 분석 결과 출력
#     print(f"\n=== Gaussian Posterior 분석 결과 ===")
#     print(f"원본 데이터 차원: {z_array.shape}")
#     print(f"PCA 설명 분산 (처음 5개): {pca.explained_variance_ratio_[:5]}")
#     print(f"PCA 누적 설명 분산 (처음 5개): {np.cumsum(pca.explained_variance_ratio_[:5])}")
#     print(f"최적 GMM 클러스터 수: {optimal_n_components}")
#     print(f"GMM BIC 점수: {gmm_optimal.bic(z_array):.2f}")
#     print(f"각 차원별 분산: {np.var(z_array, axis=0)[:10]}")  # 처음 10개 차원
#
#     # 클러스터별 특성 분석
#     print(f"\n=== 클러스터별 분석 ===")
#     for cluster_id in np.unique(cluster_labels):
#         mask = cluster_labels == cluster_id
#         flow_mean = np.mean(flow_shop_index_list[mask])
#         flow_std = np.std(flow_shop_index_list[mask])
#         bottleneck_mean = np.mean(bottleneck_index_list[mask])
#         bottleneck_std = np.std(bottleneck_index_list[mask])
#         print(f"Cluster {cluster_id} (n={np.sum(mask)}):")
#         print(f"  Flow Shop Index: {flow_mean:.3f} ± {flow_std:.3f}")
#         print(f"  Bottleneck Index: {bottleneck_mean:.3f} ± {bottleneck_std:.3f}")
#
#     print(f"\nFlow Shop과 Bottleneck Index 상관계수: {np.corrcoef(flow_shop_index_list, bottleneck_index_list)[0, 1]:.3f}")
#
#     return z_array, flow_shop_index_list, bottleneck_index_list, cluster_labels
#
def visualize_model(params, selected_param, log_path=None):
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    date = datetime.now().strftime('%m%d_%H_%M')
    param_path = params["log_dir"] + '/ppo' + '/%s_%s_param.csv' % (date, "train")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))

    epoch = 0
    ave_act_loss = 0.01
    ave_cri_loss = 0.0

    act_model = PtrNet1(params).to(device)
    baseline_model = PtrNet1(params).to(device)  # baseline_model 불필요
    baseline_model.load_state_dict(act_model.state_dict())  # baseline_model 불필요
    file_name = 'seperation_after_rep_80000_param1_step_92601_mean_makespan_3180.0'  ###########fdfdf
    checkpoint = torch.load('{}.pt'.format(file_name))
    act_model.load_state_dict(checkpoint['model_state_dict_actor'])

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    import matplotlib.colors as mcolors
    import umap  # UMAP 추가

    t1 = time()
    ave_makespan = 0

    c_max = list()
    b = 0
    problem_list = [1, 2]

    """
    변수별 shape
    inputs : batch_size X number_of_blocks X number_of_process
    pred_seq : batch_size X number_of_blocks
    """
    b += 1
    # flow_shop_index_list = list()
    # bottleneck_index_list = list()
    # makespan_list = list()
    # h_list = list()
    # z_list = list()
    # mean_feature_list = list()
    # z_mean_post_list = list()
    # machine_number_list = list()  # 추가: machine number
    # job_number_list = list()  # 추가: job number
    # machine_job_product_list = list()  # 추가: machine number × job number
    # act_model.eval()
    # for i in range(20000):
    #     num_machine = np.random.randint(5, 10)
    #     num_job = np.random.randint(num_machine, 10)
    #     jobs_datas, scheduler_list = generate_jssp_instance(num_jobs=num_job, num_machine=num_machine, batch_size=1)
    #     act_model.Latent.current_num_edges = num_machine * num_job
    #     flow_shop_index = calculate_flow_shop_index(scheduler_list[0])
    #     bottleneck_index = calculate_bottleneck_index(scheduler_list[0])
    #     machine_job_product = num_machine * num_job  # 추가: machine × job 계산
    #     act_model.get_jssp_instance(scheduler_list)
    #     scheduler = AdaptiveScheduler(jobs_datas[0])
    #     node_features = [scheduler.get_node_feature()]
    #     edge_precedence = scheduler.get_edge_index_precedence()
    #     edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
    #     edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
    #     heterogeneous_edges = [(edge_precedence, edge_antiprecedence, edge_machine_sharing)]  # 세종류의 엣지들을 하나의 변수로 참조시킴
    #     input_data = (node_features, heterogeneous_edges)
    #     z, v, h, pred_seq, mean_feature, z_mean_post = act_model.forward_visualize(input_data,
    #                                                                                device,
    #                                                                                scheduler_list=scheduler_list,
    #                                                                                num_machine=num_machine,
    #                                                                                num_job=num_job
    #                                                                                )
    #
    #     sequence = pred_seq[0]
    #     scheduler = AdaptiveScheduler(jobs_datas[0])
    #     makespan = -scheduler.run(sequence.tolist())
    #
    #     # print(z.shape)
    #     flow_shop_index_list.append(flow_shop_index)
    #     makespan_list.append(makespan)
    #     bottleneck_index_list.append(bottleneck_index)
    #     machine_number_list.append(num_machine)  # 추가
    #     job_number_list.append(num_job)  # 추가
    #     machine_job_product_list.append(machine_job_product)  # 추가
    #     h_list.append(h.squeeze(0).detach().cpu().numpy().tolist())
    #     mean_feature_list.append(mean_feature.squeeze(0).detach().cpu().numpy().tolist())
    #     z_mean_post_list.append(z_mean_post.squeeze(0).detach().cpu().numpy().tolist())
    #     z_list.append(z.squeeze(0).detach().cpu().numpy().tolist())
    #     print("data collecting : ", i)
    #
    # # numpy 배열로 변환
    # makespan_list = np.array(makespan_list)
    # flow_shop_index_list = np.array(flow_shop_index_list)
    # bottleneck_index_list = np.array(bottleneck_index_list)
    # machine_number_list = np.array(machine_number_list)  # 추가
    # job_number_list = np.array(job_number_list)  # 추가
    # machine_job_product_list = np.array(machine_job_product_list)  # 추가
    # h_array = np.array(h_list)  # z_list를 빈 리스트로 재정의하지 않고 numpy 배열로 변환
    # z_array = np.array(z_list)  # z_list를 빈 리스트로 재정의하지 않고 numpy 배열로 변환
    # mean_feature_array = np.array(mean_feature_list)
    # z_mean_post_array = np.array(z_mean_post_list)
    # makespan_list = np.array(makespan_list)
    #
    # np.save('flow_shop_index_list.npy', flow_shop_index_list)
    # np.save('bottleneck_index_list.npy', bottleneck_index_list)
    # np.save('machine_number_list.npy', machine_number_list)  # 추가
    # np.save('job_number_list.npy', job_number_list)  # 추가
    # np.save('machine_job_product_list.npy', machine_job_product_list)  # 추가
    # np.save('h_array.npy', h_array)
    # np.save('z_array.npy', z_array)
    # np.save('mean_feature_array.npy', mean_feature_array)
    # np.save('z_mean_post_array.npy', z_mean_post_array)
    # np.save('makespan_array.npy', makespan_list)
    #
    # print(f"z_array shape: {z_array.shape}")
    # print(f"flow_shop_index_list shape: {flow_shop_index_list.shape}")
    # print(f"bottleneck_index_list shape: {bottleneck_index_list.shape}")


    machine_job_product_list = np.load('machine_job_product_list.npy')  # 추가

    job_number_list = np.load('job_number_list.npy')  # 추가
    flow_shop_index_list = np.load('flow_shop_index_list.npy')#np.log( np.load('flow_shop_index_list.npy')*job_number_list/machine_job_product_list)
    bottleneck_index_list = np.load('flow_shop_index_list.npy')*job_number_list/machine_job_product_list#np.log( np.load('flow_shop_index_list.npy')/(job_number_list*machine_job_product_list))
    machine_number_list = np.load('machine_number_list.npy')  # 추가
    job_number_list = np.load('job_number_list.npy')  # 추가
    machine_job_product_list = np.load('machine_job_product_list.npy')  # 추가
    makespan_list = -np.load('makespan_array.npy')
    z_array = np.load('z_mean_post_array.npy')

    # 데이터가 비어있는지 확인
    if z_array.size == 0:
        print("Error: z_array is empty!")
        return

    scaler = StandardScaler()
    # StandardScaler 적용
    z_scaled = scaler.fit_transform(z_array)
    print(makespan_list.min(), makespan_list.max() + 300)
    # UMAP으로 2차원 축소
    print("UMAP 차원 축소 진행 중...")
    umap_reducer = umap.UMAP(n_components=5, random_state=42, n_neighbors=5, min_dist=0.5, metric='cosine')
    z_2d_umap = umap_reducer.fit_transform(z_scaled)

    # 컬러맵 범위 설정 (원하는 값으로 조절하세요)
    flow_shop_vmin, flow_shop_vmax = flow_shop_index_list.min() + 0.15, flow_shop_index_list.max() - 0.15  # Flow Shop Index 범위
    bottleneck_vmin, bottleneck_vmax = bottleneck_index_list.min(), bottleneck_index_list.max()   # Bottleneck Index 범위

    makespan_vmin, makespan_vmax = makespan_list.min() + 100, makespan_list.max() - 100  # Makespan 범위 (전체 범위 사용)
    machine_vmin, machine_vmax = 5, 10  # Machine Number 범위
    job_vmin, job_vmax = 5, 10  # Job Number 범위
    product_vmin, product_vmax = 25, 100  # Machine × Job Product 범위

    # 시각화 (UMAP만 사용, 6개의 지표)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Flow Shop Index로 컬러맵 (UMAP)
    scatter1 = axes[0, 0].scatter(z_2d_umap[:, 0], z_2d_umap[:, 1],
                                  c=flow_shop_index_list,
                                  cmap='RdBu',
                                  alpha=1.0,
                                  s=20,
                                  vmin=flow_shop_vmin,
                                  vmax=flow_shop_vmax)
    axes[0, 0].set_title('UMAP: Flow Shop Index', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('UMAP Component 1')
    axes[0, 0].set_ylabel('UMAP Component 2')
    axes[0, 0].grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter1, ax=axes[0, 0])
    cbar1.set_label('Flow Shop Index')

    # 2. Bottleneck Index로 컬러맵 (UMAP)
    scatter2 = axes[0, 1].scatter(z_2d_umap[:, 0], z_2d_umap[:, 1],
                                  c=bottleneck_index_list,
                                  cmap='RdBu',
                                  alpha=1.0,
                                  s=20,
                                  vmin=bottleneck_vmin,
                                  vmax=bottleneck_vmax)
    axes[0, 1].set_title('UMAP: Bottleneck Index', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('UMAP Component 1')
    axes[0, 1].set_ylabel('UMAP Component 2')
    axes[0, 1].grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=axes[0, 1])
    cbar2.set_label('Bottleneck Index')

    # 3. Makespan으로 컬러맵 (UMAP)
    scatter3 = axes[0, 2].scatter(z_2d_umap[:, 0], z_2d_umap[:, 1],
                                  c=makespan_list,
                                  cmap='RdBu',
                                  alpha=1.0,
                                  s=20,
                                  vmin=makespan_vmin,
                                  vmax=makespan_vmax)
    axes[0, 2].set_title('UMAP: Makespan', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('UMAP Component 1')
    axes[0, 2].set_ylabel('UMAP Component 2')
    axes[0, 2].grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=axes[0, 2])
    cbar3.set_label('Makespan')

    # 4. Machine Number로 컬러맵 (UMAP) - 새로 추가
    scatter4 = axes[1, 0].scatter(z_2d_umap[:, 0], z_2d_umap[:, 1],
                                  c=machine_number_list,
                                  cmap='RdBu',
                                  alpha=1.0,
                                  s=20,
                                  vmin=machine_vmin,
                                  vmax=machine_vmax)
    axes[1, 0].set_title('UMAP: Machine Number', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('UMAP Component 1')
    axes[1, 0].set_ylabel('UMAP Component 2')
    axes[1, 0].grid(True, alpha=0.3)
    cbar4 = plt.colorbar(scatter4, ax=axes[1, 0])
    cbar4.set_label('Machine Number')

    # 5. Job Number로 컬러맵 (UMAP) - 새로 추가
    scatter5 = axes[1, 1].scatter(z_2d_umap[:, 0], z_2d_umap[:, 1],
                                  c=job_number_list,
                                  cmap='RdBu',
                                  alpha=1.0,
                                  s=20,
                                  vmin=job_vmin,
                                  vmax=job_vmax)
    axes[1, 1].set_title('UMAP: Job Number', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('UMAP Component 1')
    axes[1, 1].set_ylabel('UMAP Component 2')
    axes[1, 1].grid(True, alpha=0.3)
    cbar5 = plt.colorbar(scatter5, ax=axes[1, 1])
    cbar5.set_label('Job Number')

    # 6. Machine × Job Product로 컬러맵 (UMAP)
    scatter6 = axes[1, 2].scatter(z_2d_umap[:, 0], z_2d_umap[:, 1],
                                  c=machine_job_product_list,
                                  cmap='RdBu',
                                  alpha=1.0,
                                  s=20,
                                  vmin=product_vmin,
                                  vmax=product_vmax)
    axes[1, 2].set_title('UMAP: Machine × Job Product', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('UMAP Component 1')
    axes[1, 2].set_ylabel('UMAP Component 2')
    axes[1, 2].grid(True, alpha=0.3)
    cbar6 = plt.colorbar(scatter6, ax=axes[1, 2])
    cbar6.set_label('Machine × Job Product')

    plt.tight_layout()
    plt.show()

    # 상관관계 분석
    print(f"\n=== 데이터 분석 결과 ===")
    print(f"원본 데이터 차원: {z_array.shape}")
    print(f"UMAP 축소된 데이터 차원: {z_2d_umap.shape}")
    print(f"Flow Shop Index 범위: {flow_shop_index_list.min():.2f} ~ {flow_shop_index_list.max():.2f}")
    print(f"Bottleneck Index 범위: {bottleneck_index_list.min():.2f} ~ {bottleneck_index_list.max():.2f}")
    print(f"Makespan 범위: {makespan_list.min():.2f} ~ {makespan_list.max():.2f}")
    print(f"Machine Number 범위: {machine_number_list.min()} ~ {machine_number_list.max()}")
    print(f"Job Number 범위: {job_number_list.min()} ~ {job_number_list.max()}")
    print(f"Machine × Job Product 범위: {machine_job_product_list.min()} ~ {machine_job_product_list.max()}")
    print(f"Flow Shop과 Bottleneck Index 상관계수: {np.corrcoef(flow_shop_index_list, bottleneck_index_list)[0, 1]:.3f}")
    print(f"Flow Shop과 Makespan 상관계수: {np.corrcoef(flow_shop_index_list, makespan_list)[0, 1]:.3f}")
    print(f"Flow Shop과 Machine Number 상관계수: {np.corrcoef(flow_shop_index_list, machine_number_list)[0, 1]:.3f}")
    print(f"Flow Shop과 Job Number 상관계수: {np.corrcoef(flow_shop_index_list, job_number_list)[0, 1]:.3f}")
    print(
        f"Flow Shop과 Machine×Job Product 상관계수: {np.corrcoef(flow_shop_index_list, machine_job_product_list)[0, 1]:.3f}")
    print(f"Bottleneck Index와 Makespan 상관계수: {np.corrcoef(bottleneck_index_list, makespan_list)[0, 1]:.3f}")
    print(f"Bottleneck Index와 Machine Number 상관계수: {np.corrcoef(bottleneck_index_list, machine_number_list)[0, 1]:.3f}")
    print(f"Bottleneck Index와 Job Number 상관계수: {np.corrcoef(bottleneck_index_list, job_number_list)[0, 1]:.3f}")
    print(
        f"Bottleneck Index와 Machine×Job Product 상관계수: {np.corrcoef(bottleneck_index_list, machine_job_product_list)[0, 1]:.3f}")
    print(f"Makespan과 Machine Number 상관계수: {np.corrcoef(makespan_list, machine_number_list)[0, 1]:.3f}")
    print(f"Makespan과 Job Number 상관계수: {np.corrcoef(makespan_list, job_number_list)[0, 1]:.3f}")
    print(f"Makespan과 Machine×Job Product 상관계수: {np.corrcoef(makespan_list, machine_job_product_list)[0, 1]:.3f}")
    print(f"Machine Number과 Job Number 상관계수: {np.corrcoef(machine_number_list, job_number_list)[0, 1]:.3f}")
    print(
        f"Machine Number과 Machine×Job Product 상관계수: {np.corrcoef(machine_number_list, machine_job_product_list)[0, 1]:.3f}")
    print(f"Job Number과 Machine×Job Product 상관계수: {np.corrcoef(job_number_list, machine_job_product_list)[0, 1]:.3f}")

    # 추가 시각화: 히스토그램 (6개 지표)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    axes[0, 0].hist(flow_shop_index_list, bins=30, alpha=1.0, color='green', edgecolor='black')
    axes[0, 0].set_title('Flow Shop Index 분포')
    axes[0, 0].set_xlabel('Flow Shop Index')
    axes[0, 0].set_ylabel('빈도')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(bottleneck_index_list, bins=30, alpha=1.0, color='purple', edgecolor='black')
    axes[0, 1].set_title('Bottleneck Index 분포')
    axes[0, 1].set_xlabel('Bottleneck Index')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].hist(makespan_list, bins=30, alpha=1.0, color='orange', edgecolor='black')
    axes[0, 2].set_title('Makespan 분포')
    axes[0, 2].set_xlabel('Makespan')
    axes[0, 2].set_ylabel('빈도')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].hist(machine_number_list, bins=range(5, 11), alpha=1.0, color='blue', edgecolor='black')
    axes[1, 0].set_title('Machine Number 분포')
    axes[1, 0].set_xlabel('Machine Number')
    axes[1, 0].set_ylabel('빈도')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(job_number_list, bins=range(5, 11), alpha=1.0, color='cyan', edgecolor='black')
    axes[1, 1].set_title('Job Number 분포')
    axes[1, 1].set_xlabel('Job Number')
    axes[1, 1].set_ylabel('빈도')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(machine_job_product_list, bins=30, alpha=1.0, color='red', edgecolor='black')
    axes[1, 2].set_title('Machine × Job Product 분포')
    axes[1, 2].set_xlabel('Machine × Job Product')
    axes[1, 2].set_ylabel('빈도')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
