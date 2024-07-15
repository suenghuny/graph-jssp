import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from time import time
from datetime import datetime
from actor2 import PtrNet1
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
set_seed(20)  # 30 했었음
opt_list = [1059, 888, 1005, 1005, 887, 1010, 397, 899, 934, 944]
orb_list = []
problem_number =  ['71','72','61','62','51','52','41','42','31'
    ,'32','21','22']
for i in problem_number:
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
    pred_seq, ll_old, _ = act_model(input_data,
                                    device,
                                    scheduler_list=scheduler_list_val,
                                    num_machine=num_machine,
                                    num_job=num_job,
                                    upperbound=upperbound)
    makespan_list = list()
    j = 0
    for sequence in pred_seq:
        scheduler = AdaptiveScheduler(orb_list[p - 1])
        makespan = scheduler.run(sequence.tolist())
        val_makespan.append(makespan)
        makespan_list.append(makespan)
        print(j, makespan)
        j+=1
    # print("크크크", val_makespan)
    return np.min(val_makespan), np.mean(val_makespan), makespan_list


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

    checkpoint = torch.load(params["model_dir"] +
                            '/ppo_w_third_feature/' +
                            '0713_20_37_step48401_act.pt')
    # new 17101 O
    # 17201 <
    # 16901 X
    # 16801 X
    # 23101 X
    # 26501 O 31501
    # 29501 괜찮음
    # 29201 O
    # 40401 ㅒ
    # 39501 X
    # 42401 <
    # 48401 <


    # New27801 O
    # 42301 X
    # 42201 >
    # 48201 >
    # 48101 X
    # 48401 X
    # 49401 >
    # 49701 >
    # 49201 >

    #checkpoint = torch.load(params["model_dir"] + '/ppo_w_third_feature/' + '0613_19_18_step20801_act.pt')
    #20001도 괘아늠
    #20751도 괘아늠
    #20251 괘아늠
    #20651 별로

    act_model.load_state_dict(checkpoint['model_state_dict_actor'])

    baseline_model = PtrNet1(params).to(device)  # baseline_model 불필요
    baseline_model.load_state_dict(act_model.state_dict())  # baseline_model 불필요
    if params["optimizer"] == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=params["lr"])
        """
        act_model이라는 신경망 뭉치에 파라미터(가중치, 편향)을 업데이트 할꺼야.

        """

    elif params["optimizer"] == "RMSProp":
        act_optim = optim.RMSprop(act_model.parameters(), lr=params["lr"])
    if params["is_lr_decay"]:
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])

    t1 = time()
    ave_makespan = 0

    c_max = list()
    b = 0
    problem_list = [1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    validation_records_min = [[] for _ in problem_list]
    validation_records_mean = [[] for _ in problem_list]

    for s in range(epoch + 1, params["step"]):

        """

        변수별 shape 
        inputs : batch_size X number_of_blocks X number_of_process
        pred_seq : batch_size X number_of_blocks

        """
        b += 1

        if s % 100 == 1:  # Evaluation 수행
            makespan_records = dict()
            for p in problem_list:
                min_makespan = heuristic_eval(p)
                eval_number = 2
                with torch.no_grad():
                    min_makespan_list = [min_makespan] * eval_number
                    min_makespan1, mean_makespan1, makespan_list1 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan2, mean_makespan2, makespan_list2 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan3, mean_makespan3, makespan_list3 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan4, mean_makespan4, makespan_list4 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan5, mean_makespan5, makespan_list5 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan6, mean_makespan6, makespan_list6 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan7, mean_makespan7, makespan_list7 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan8, mean_makespan8, makespan_list8 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan9, mean_makespan9, makespan_list9 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan10, mean_makespan10, makespan_list10 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan11, mean_makespan11, makespan_list11 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan12, mean_makespan12, makespan_list12 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan13, mean_makespan13, makespan_list13 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan14, mean_makespan14, makespan_list14 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                    min_makespan15, mean_makespan15, makespan_list15 = evaluation(act_model, baseline_model, p, eval_number, device, upperbound=min_makespan_list)
                min_makespan = np.min([min_makespan1, min_makespan2, min_makespan3, min_makespan4, min_makespan5, min_makespan6, min_makespan7,
                                       min_makespan8, min_makespan9, min_makespan10, min_makespan11, min_makespan12, min_makespan13, min_makespan14, min_makespan15])
                mean_makespan = (mean_makespan1 + mean_makespan2 + mean_makespan3 + mean_makespan4 + mean_makespan5 + mean_makespan6+ mean_makespan7+ mean_makespan8+ mean_makespan9+ mean_makespan10+ mean_makespan11+ mean_makespan12+ mean_makespan13+ mean_makespan14+ mean_makespan15) / 15
                makespan_list = [makespan for makespan in makespan_list1]+\
                                [makespan for makespan in makespan_list2]+ \
                                [makespan for makespan in makespan_list3] + \
                                [makespan for makespan in makespan_list4] + \
                                [makespan for makespan in makespan_list5] + \
                                [makespan for makespan in makespan_list6] + \
                                [makespan for makespan in makespan_list7]+ \
                                [makespan for makespan in makespan_list8] + \
                                [makespan for makespan in makespan_list9] + \
                                [makespan for makespan in makespan_list10] + \
                                [makespan for makespan in makespan_list11] + \
                                [makespan for makespan in makespan_list12] + \
                                [makespan for makespan in makespan_list13] + \
                                [makespan for makespan in makespan_list14] + \
                                [makespan for makespan in makespan_list15]
                name = problem_number[p-1]
                j = 0
                for makespan in makespan_list:
                    makespan_records[name+str(j)] = makespan
                    j+=1
                print(makespan_records)
                df = pd.DataFrame(list(makespan_records.items()), columns=['Key', 'Value'])
                df.to_csv("makespan_records.csv")

                print("TA{}".format(problem_list[p - 1]), min_makespan, mean_makespan)


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
                    #min_m.to_csv('min_makespan_w_third_feature2.csv')
                    #mean_m.to_csv('mean_makespan_w_third_feature2.csv')

        act_model.block_indices = []
        baseline_model.block_indices = []

        if s % cfg.gen_step == 1:  # 훈련용 Data Instance 새롭게 생성 (gen_step 마다 생성)
            """
            훈련용 데이터셋 생성하는 코드
            """
            num_job = np.random.randint(5, 10)
            num_machine = np.random.randint(num_job, 10)
            jobs_datas, scheduler_list = generate_jssp_instance(num_jobs=num_job, num_machine=num_machine,
                                                                batch_size=params['batch_size'])
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
            pred_seq, ll_old, _ = act_model(input_data,
                                            device,
                                            scheduler_list=scheduler_list,
                                            num_machine=num_machine,
                                            num_job=num_job,
                                            upperbound=makespan_list_for_upperbound
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
            beta = params['beta']
            if baseline_reset == False:
                if s == 1:
                    be = torch.tensor(real_makespan).detach().unsqueeze(1).to(device)  # baseline을 구하는 부분
                else:
                    be = beta * be + (1 - beta) * torch.tensor(real_makespan).to(device)
            else:
                if s % cfg.gen_step == 1:
                    be = torch.tensor(real_makespan).detach().unsqueeze(1).to(device)  # baseline을 구하는 부분
                else:
                    be = beta * be + (1 - beta) * torch.tensor(real_makespan).unsqueeze(1).to(device)
            ####

            act_optim.zero_grad()
            adv = torch.tensor(real_makespan).detach().unsqueeze(1).to(device) - be  # baseline(advantage) 구하는 부분
            """

            1. Loss 구하기
            2. Gradient 구하기 (loss.backward)
            3. Update 하기(act_optim.step)

            """
            act_loss = -(ll_old * adv).mean()  # loss 구하는 부분 /  ll_old의 의미 log_theta (pi | s)
            act_loss.backward()
            nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=float(os.environ.get("grad_clip", 10)),
                                     norm_type=2)
            act_optim.step()
        if cfg.algo == 'ppo':
            pred_seq, ll_old, old_sequence = act_model(input_data,
                                                       device,
                                                       scheduler_list=scheduler_list,
                                                       num_machine=num_machine,
                                                       num_job=num_job,
                                                       upperbound=makespan_list_for_upperbound
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
            beta = params['beta']
            if baseline_reset == False:
                if s == 1:
                    be = torch.tensor(real_makespan).detach().unsqueeze(1).to(device)  # baseline을 구하는 부분
                else:
                    be = beta * be + (1 - beta) * torch.tensor(real_makespan).to(device)
            else:
                if s % cfg.gen_step == 1:
                    be = torch.tensor(real_makespan).detach().unsqueeze(1).to(device)  # baseline을 구하는 부분
                else:
                    be = beta * be + (1 - beta) * torch.tensor(real_makespan).unsqueeze(1).to(device)
            ####
            for i in range(params['k_epoch']):
                for scheduler in scheduler_list:
                    scheduler.reset()
                _, ll_new, _ = act_model(input_data,
                                         device,
                                         scheduler_list=scheduler_list,
                                         num_machine=num_machine,
                                         num_job=num_job,
                                         upperbound=makespan_list_for_upperbound,
                                         old_sequence=old_sequence
                                         )

                ratio = torch.exp(ll_new - ll_old.detach()).unsqueeze(-1)
                adv = torch.tensor(real_makespan).detach().unsqueeze(1).to(device) - be  # baseline(advantage) 구하는 부분

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - params["epsilon"], 1 + params["epsilon"]) * adv
                act_loss = -torch.min(surr1, surr2).mean()
                act_optim.zero_grad()
                act_loss.backward()
                nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=float(os.environ.get("grad_clip", 10)),
                                         norm_type=2)
                act_optim.step()

        if act_lr_scheduler.get_last_lr()[0] >= 1e-4:
            if params["is_lr_decay"]:
                act_lr_scheduler.step()
        ave_act_loss += act_loss.item()

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
            if vessl == False:
                torch.save({'epoch': s,
                            'model_state_dict_actor': act_model.state_dict(),
                            'optimizer_state_dict_actor': act_optim.state_dict(),
                            'ave_act_loss': ave_act_loss,
                            'ave_cri_loss': 0,
                            'ave_makespan': ave_makespan},
                           params["model_dir"] + '/ppo_w_third_feature' + '/%s_step%d_act.pt' % (date, s))
            else:
                torch.save({'epoch': s,
                            'model_state_dict_actor': act_model.state_dict(),
                            'optimizer_state_dict_actor': act_optim.state_dict(),
                            'ave_act_loss': ave_act_loss,
                            'ave_cri_loss': 0,
                            'ave_makespan': ave_makespan},
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
        "beta": float(os.environ.get("beta", 0.6)),
        "alpha": float(os.environ.get("alpha", 0.1)),
        "lr": float(os.environ.get("lr", 1.0e-3)),
        "lr_decay": float(os.environ.get("lr_decay", 0.95)),
        "lr_decay_step":
            int(os.environ.get("lr_decay_step", 700)),
        "layers": eval(str(os.environ.get("layers", '[196, 96]'))),
        "n_embedding":
            int(os.environ.get("n_embedding", 48)),
        "n_hidden": int(os.environ.get("n_hidden", 84)),
        "graph_embedding_size": int(os.environ.get("graph_embedding_size", 100)),
        "n_multi_head": int(os.environ.get("n_multi_head", 2)),
        "ex_embedding_size": int(os.environ.get("ex_embedding_size", 40)),
        "k_hop": int(os.environ.get("k_hop", 1)),
        "is_lr_decay": True,
        "third_feature": 'first_and_second',  # first_and_second, first_only, second_only
        "baseline_reset": True,
        "ex_embedding": True,
        "k_epoch": int(os.environ.get("k_epoch", 2)),

    }

    train_model(params)  #