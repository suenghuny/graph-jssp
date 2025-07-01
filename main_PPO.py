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
for i in ["61", "62"]:
    df = pd.read_excel("ta.xlsx", sheet_name=i, engine='openpyxl')
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
                                    train = False)
    for sequence in pred_seq:
        scheduler = AdaptiveScheduler(orb_list[p - 1])
        makespan = scheduler.run(sequence.tolist())
        val_makespan.append(makespan)
    # print("크크크", val_makespan)
    return np.min(val_makespan), np.mean(val_makespan)
#0628_20_59_step86000_act_w_rep

def train_model(params, selected_param, log_path=None):
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

    baseline_model = PtrNet1(params).to(device)  # baseline_model 불필요
    baseline_model.load_state_dict(act_model.state_dict())  # baseline_model 불필요
    if params["optimizer"] == 'Adam':
        latent_optim = optim.Adam(act_model.Latent.parameters(), lr=params["lr_latent"])
        act_optim = optim.Adam(act_model.all_attention_params, lr=params["lr_critic"])
        cri_optim = optim.Adam(act_model.critic.parameters(), lr=params['lr'])
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"], gamma=params["lr_decay"])
        cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, step_size=params["lr_decay_step"], gamma=params["lr_decay"])
        latent_lr_scheduler = optim.lr_scheduler.StepLR(latent_optim, step_size=params["lr_decay_step"], gamma=params["lr_decay"])
        entropy_coeff_optim = optim.Adam([act_model.log_alpha], 1e-5)
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
                eval_number = 5
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
                        min_m.to_csv('w_rep_min_makespan_{}.csv'.format(selected_param))
                        mean_m.to_csv('w_rep_mean_makespan_{}.csv'.format(selected_param))
                    else:
                        min_m.to_csv('wo_rep_min_makespan_{}.csv'.format(selected_param))
                        mean_m.to_csv('wo_rep_mean_makespan_{}.csv'.format(selected_param))
            wandb.log({
                "episode": s,
                "71 mean_makespan": mean_makespan71,
                "72 mean_makespan": mean_makespan72,
                "71 min_makespan": min_makespan71,
                "72 min_makespan": min_makespan72,

            })
        act_model.block_indices = []
        baseline_model.block_indices = []

        if s % cfg.gen_step == 1:  # 훈련용 Data Instance 새롭게 생성 (gen_step 마다 생성)
            """
            훈련용 데이터셋 생성하는 코드
            """
            num_machine = np.random.randint(5, 10)
            num_job = np.random.randint(num_machine, 10)


            jobs_datas, scheduler_list = generate_jssp_instance(num_jobs=num_job, num_machine=num_machine,
                                                                batch_size=params['batch_size'])
            act_model.Latent.current_num_edges = num_machine*num_job

            # print(jobs_datas)
            # print("======================")
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
            entropy_loss = params['entropy_coeff'] * entropy
            # target_entropy = params["target_entropy"]
            # log_alpha_loss = -act_model.log_alpha * (ll_old.detach() + target_entropy).mean()
            adv = torch.tensor(real_makespan).detach().unsqueeze(1).to(device) - baselines  # baseline(advantage) 구하는 부분
            cri_loss = F.mse_loss(torch.tensor(real_makespan).to(device)+entropy_loss.detach().squeeze(1), baselines.squeeze(1))

            """

            1. Loss 구하기
            2. Gradient 구하기 (loss.backward)
            3. Update 하기(act_optim.step)

            """
            latent_loss = edge_loss+node_loss+loss_kld
            act_loss = -(ll_old * adv.detach()+entropy_loss).mean()  # loss 구하는 부분 /  ll_old의 의미 log_theta (pi | s)

            if params['w_representation_learning'] == True:
                total_loss = latent_loss + act_loss + cri_loss#+log_alpha_loss
            else:
                total_loss = act_loss + cri_loss
            latent_optim.zero_grad()
            act_optim.zero_grad()
            cri_optim.zero_grad()
            entropy_coeff_optim.zero_grad()
            total_loss.backward()

            #print("critic loss : ", np.round(cri_loss.detach().cpu().numpy().tolist(), 2), " q : ", np.round(q.detach().mean().cpu().numpy().tolist(), 2), " act loss : ",np.round(act_loss.detach().cpu().numpy().tolist(), 2), " sample makespan : ", np.round(torch.tensor(sampled_makespans).float().to(device).mean().detach().cpu().numpy().tolist(), 2))




            nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=float(os.environ.get("grad_clip", 5)), norm_type=2)



            # 그 후 각 옵티마이저 단계 실행
            if s <=20000:
                latent_optim.step()
            act_optim.step()
            cri_optim.step()
            step_with_min(act_lr_scheduler, act_optim, min_lr=params['lr_decay_min'])
            step_with_min(cri_lr_scheduler, cri_optim, min_lr=params['lr_decay_min'])

            if params['w_representation_learning'] == True:
                if latent_lr_scheduler.get_last_lr()[0] >=  float(os.environ.get("lr_decay_min", 5.0e-5)):
                    latent_lr_scheduler.step()


        ave_act_loss += act_loss.item()
        if s % params["log_step"] == 0:
            t2 = time()
            if params['w_representation_learning'] == True:
                print('with representation learning step:%d/%d, actic loss:%1.3f, crictic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                    s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]),
                    ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan, (t2 - t1) // 60, (t2 - t1) % 60))
            else:
                print('without representation learning step:%d/%d, actic loss:%1.3f, crictic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                    s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]),
                    ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan, (t2 - t1) // 60, (t2 - t1) % 60))

        if mean_makespan72<=1910:
            if params['w_representation_learning'] == True:
                torch.save({'epoch': s,
                            'model_state_dict_actor': act_model.state_dict(),
                            'optimizer_state_dict_actor': act_optim.state_dict(),
                            'ave_act_loss': ave_act_loss,
                            'ave_cri_loss': 0,
                            'ave_makespan': ave_makespan},
                           params["model_dir"] + '/%s_step%d_act_w_rep.pt' % (date, s))
            else:
                torch.save({'epoch': s,
                            'model_state_dict_actor': act_model.state_dict(),
                            'optimizer_state_dict_actor': act_optim.state_dict(),
                            'ave_act_loss': ave_act_loss,
                            'ave_cri_loss': 0,
                            'ave_makespan': ave_makespan},
                           params["model_dir"] + '/%s_step%d_act_wo_rep.pt' % (date, s))



def test_model(params, selected_param, log_path=None):
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
    baseline_model = PtrNet1(params).to(device)  # baseline_model 불필요
    baseline_model.load_state_dict(act_model.state_dict())  # baseline_model 불필요
    file_name = '0628_20_59_step87300_act_w_rep'
    checkpoint = torch.load('result/model/' + '{}.pt'.format(file_name))
    act_model.load_state_dict(checkpoint['model_state_dict_actor'])


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
                eval_number = 5
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
                        min_m.to_csv('w_rep_min_makespan_{}.csv'.format(selected_param))
                        mean_m.to_csv('w_rep_mean_makespan_{}.csv'.format(selected_param))
                    else:
                        min_m.to_csv('wo_rep_min_makespan_{}.csv'.format(selected_param))
                        mean_m.to_csv('wo_rep_mean_makespan_{}.csv'.format(selected_param))
            wandb.log({
                "episode": s,
                "71 mean_makespan": mean_makespan71,
                "72 mean_makespan": mean_makespan72,
                "71 min_makespan": min_makespan71,
                "72 min_makespan": min_makespan72,

            })



#
#
# if __name__ == '__main__':
#
#     load_model = False
#
#     log_dir = "./result/log"
#     if not os.path.exists(log_dir + "/ppo"):
#         os.makedirs(log_dir + "/ppo")
#
#     model_dir = "./result/model"
#     if not os.path.exists(model_dir + "/ppo_w_third_feature"):
#         os.makedirs(model_dir + "/ppo_w_third_feature")
#
#     param_group = 1
#     params = {
#         "num_of_process": 6,
#         "step": cfg.step,
#         "log_step": cfg.log_step,
#         "log_dir": log_dir,
#         "save_step": cfg.save_step,
#         "model_dir": model_dir,
#         "batch_size": cfg.batch_size,
#         "init_min": -0.08,
#         "init_max": 0.08,
#         "use_logit_clipping": True,
#         "C": cfg.C,
#         "T": cfg.T,
#         "iteration": cfg.iteration,
#         "epsilon": float(os.environ.get("epsilon", 0.2)),
#         "optimizer": "Adam",
#         "n_glimpse": cfg.n_glimpse,
#         "n_process": cfg.n_process,
#         "lr_decay_step_critic": cfg.lr_decay_step_critic,
#         "load_model": load_model,
#         "entropy_coeff": float(os.environ.get("entropy_loss", 0.00001)),
#         "dot_product": cfg.dot_product,
#         "lr_critic": cfg.lr_critic,
#
#         "reward_scaler": cfg.reward_scaler,
#         "beta": float(os.environ.get("beta", 0.65)),
#         "alpha": float(os.environ.get("alpha", 0.1)),
#         "lr_latent": float(os.environ.get("lr_latent", 5.0e-5)),
#         "lr_critic": float(os.environ.get("lr_critic", 1.0e-4)),
#         "lr": float(os.environ.get("lr", 1.0e-4)),
#         "lr_decay": float(os.environ.get("lr_decay", 0.95)),
#         "lr_decay_step": int(os.environ.get("lr_decay_step",500)),
#         "layers": eval(str(os.environ.get("layers", '[256, 128]'))),
#         "n_embedding": int(os.environ.get("n_embedding", 48)),
#         "n_hidden": int(os.environ.get("n_hidden", 108)),
#         "graph_embedding_size": int(os.environ.get("graph_embedding_size", 96)),
#         "n_multi_head": int(os.environ.get("n_multi_head",2)),
#         "ex_embedding_size": int(os.environ.get("ex_embedding_size",36)),
#         "ex_embedding_size2": int(os.environ.get("ex_embedding_size2", 48)),
#         "k_hop": int(os.environ.get("k_hop", 1)),
#         "is_lr_decay": True,
#         "third_feature": 'first_and_second',  # first_and_second, first_only, second_only
#         "baseline_reset": True,
#         "ex_embedding": True,
#         "w_representation_learning":True,
#         "z_dim": 128,
#         "k_epoch": int(os.environ.get("k_epoch", 1)),
#         "target_entropy": int(os.environ.get("target_entropy", -2)),
#     }
#
#
#
#     wandb.login()
#     if params['w_representation_learning'] == True:
#         wandb.init(project="Graph JSSP", name="W_REP GES_{} EXEMB_{}_{}  KHOP_{} NMH_{} NH_{} LR_{} LR CRI_{} LR LAT_{} EC_{}".format(params['graph_embedding_size'],
#                                                                            params['ex_embedding_size'],
#                                                                            params['ex_embedding_size2'],
#                                                                            params['k_hop'],
#
#                                                                            params['n_multi_head'],
#                                                                            params['n_hidden'],
#                                                                                                                       params['lr'],
#                                                                                                                       params['lr_critic'],
#                                                                                                                                 params['lr_latent'],
#                                                                                                                                 params[
#                                                                                                                                     'entropy_coeff']
#                                                                                                                                 ))
#     else:
#         wandb.init(project="Graph JSSP", name="WO_REP GES_{} EXEMB_{}_{} KHOP_{} NMH_{} NH_{} LR_{} LR CRI_{} LR LAT_{} EC_{}".format(params['graph_embedding_size'],
#                                                                            params['ex_embedding_size'],
#                                                                            params['ex_embedding_size2'],
#                                                                            params['k_hop'],
#                                                                            params['n_multi_head'],
#                                                                            params['n_hidden'],
#                                                                                                                       params['lr'],
#                                                                                                                       params['lr_critic'],
#                                                                                                                                 params['lr_latent'],
#                                                                                                                                 params[
#                                                                                                                                     'entropy_coeff']
#                                                                                                                                 ))
#     train_model(params)