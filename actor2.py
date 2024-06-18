import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cfg
from copy import deepcopy
cfg = cfg.get_cfg()
from model import GCRN
device = torch.device(cfg.device)


class Categorical(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)


class ExEmbedding(nn.Module):
    def __init__(self, raw_feature_size, feature_size):
        super().__init__()
        self.fcn1 = nn.Linear(raw_feature_size, 84)
        self.fcn2 = nn.Linear(84, 64)
        self.fcn3 = nn.Linear(64, feature_size)
    def forward(self, x):
        x = F.elu(self.fcn1(x))
        x = F.elu(self.fcn2(x))
        x = self.fcn3(x)
        return x


class PtrNet1(nn.Module):
    def __init__(self, params):
        super().__init__()
        device = torch.device(cfg.device)
        self.n_multi_head = params["n_multi_head"]
        self.Embedding = nn.Linear(params["num_of_process"],params["n_hidden"], bias=False).to(device)  # 그림 상에서 Encoder에 FF(feedforward)라고 써져있는 부분
        self.params = params
        self.k_hop = params["k_hop"]

        num_edge_cat = 3
        self.GraphEmbedding = GCRN(feature_size =  params["n_hidden"],
                                   graph_embedding_size= params["graph_embedding_size"],
                                   embedding_size =  params["n_hidden"],
                                   layers =  params["layers"],
                                   alpha =  params["alpha"],
                                   n_multi_head = params["n_multi_head"],
                                   num_edge_cat = num_edge_cat).to(device)
        self.GraphEmbedding1 = GCRN(feature_size =  params["n_hidden"],
                                   graph_embedding_size= params["graph_embedding_size"],
                                   embedding_size =  params["n_hidden"],
                                    n_multi_head=params["n_multi_head"],
                                    layers =  params["layers"],
                                    alpha=params["alpha"],
                                    num_edge_cat = num_edge_cat).to(device)
        #print(self.k_hop, self.n_multi_head)
        # self.GraphEmbedding2 = GCRN(feature_size= params["n_hidden"],
        #                             graph_embedding_size=params["graph_embedding_size"],
        #                             embedding_size=params["n_hidden"],
        #                             layers=params["layers"],
        #                             num_edge_cat=num_edge_cat).to(device)

        # self.GraphEmbedding3 = GCRN(feature_size= params["n_hidden"],
        #                             graph_embedding_size=params["graph_embedding_size"],
        #                             embedding_size=params["n_hidden"],
        #                             layers=params["layers"],
        #                             num_edge_cat=num_edge_cat).to(device)
        # self.GraphEmbedding4 = GCRN(feature_size= params["n_hidden"],
        #                             graph_embedding_size=params["graph_embedding_size"],
        #                             embedding_size=params["n_hidden"],
        #                             layers=params["layers"],
        #                             num_edge_cat=num_edge_cat).to(device)
        # self.GraphEmbedding5 = GCRN(feature_size=params["n_hidden"],
        #                             graph_embedding_size=params["graph_embedding_size"],
        #                             embedding_size=params["n_hidden"],
        #                             layers=params["layers"],
        #                             num_edge_cat=num_edge_cat).to(device)
        # self.GraphEmbedding6 = GCRN(feature_size=params["n_hidden"],
        #                             graph_embedding_size=params["graph_embedding_size"],
        #                             embedding_size=params["n_hidden"],
        #                             layers=params["layers"],
        #                             num_edge_cat=num_edge_cat).to(device)
        # self.GraphEmbedding7 = GCRN(feature_size=params["n_hidden"],
        #                             graph_embedding_size=params["graph_embedding_size"],
        #                             embedding_size=params["n_hidden"],
        #                             layers=params["layers"],
        #                             num_edge_cat=num_edge_cat).to(device)



        augmented_hidden_size = params["n_hidden"]

        if self.params['third_feature'] == "first_and_second":
            if self.params['ex_embedding'] == True:
                extended_dimension = params["ex_embedding_size"]
                self.ex_embedding = ExEmbedding(raw_feature_size=4, feature_size = params["ex_embedding_size"])
            else:
                extended_dimension = 4
        if (self.params['third_feature'] == "first_only") or \
           (self.params['third_feature'] == "second_only"):
            if self.params['ex_embedding'] == True:
                extended_dimension = params["ex_embedding_size"]
                self.ex_embedding = ExEmbedding(raw_feature_size=2, feature_size=params["ex_embedding_size"])
            else:
                extended_dimension = 2

        self.Vec = [nn.Parameter(torch.FloatTensor(augmented_hidden_size+extended_dimension, augmented_hidden_size)) for _ in range(self.n_multi_head)]
        self.Vec = nn.ParameterList(self.Vec)
        self.W_q = [nn.Linear(2*augmented_hidden_size, augmented_hidden_size, bias=False).to(device)  for _ in range(self.n_multi_head)]
        self.W_q_weights = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q])
        self.W_q_biases = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_q])

        self.W_ref =[nn.Linear(augmented_hidden_size+extended_dimension,augmented_hidden_size, bias=False).to(device) for _ in range(self.n_multi_head)]
        self.W_ref_weights = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_ref])
        self.W_ref_biases = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_ref])

        self.Vec3 = [nn.Parameter(torch.FloatTensor(augmented_hidden_size+extended_dimension, augmented_hidden_size)) for _ in range(self.n_multi_head)]
        self.Vec3 = nn.ParameterList(self.Vec3)
        self.W_q3 = [nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False).to(device)  for _ in range(self.n_multi_head)]
        self.W_q_weights3 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q3])
        self.W_q_biases3 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_q3])
        self.W_ref3 =[nn.Linear(augmented_hidden_size+extended_dimension,augmented_hidden_size, bias=False).to(device) for _ in range(self.n_multi_head)]
        self.W_ref_weights3 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_ref3])
        self.W_ref_biases3 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_ref3])
        self.Vec4 = [nn.Parameter(torch.FloatTensor(augmented_hidden_size+extended_dimension, augmented_hidden_size)) for _ in range(self.n_multi_head)]
        self.Vec4 = nn.ParameterList(self.Vec4)
        self.W_q4 = [nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False).to(device)  for _ in range(self.n_multi_head)]
        self.W_q_weights4 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q4])
        self.W_q_biases4 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_q4])
        self.W_ref4 =[nn.Linear(augmented_hidden_size+extended_dimension,augmented_hidden_size, bias=False).to(device) for _ in range(self.n_multi_head)]
        self.W_ref_weights4 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_ref4])
        self.W_ref_biases4 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_ref4])
        self.Vec2 = nn.Parameter(torch.FloatTensor(augmented_hidden_size))
        self.W_q2 = nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False)
        self.W_ref2 = nn.Linear(augmented_hidden_size+extended_dimension,augmented_hidden_size, bias=False)
        self.dec_input = nn.Parameter(torch.FloatTensor(augmented_hidden_size))
        self.v_1 = nn.Parameter(torch.FloatTensor(augmented_hidden_size))
        self.v_f = nn.Parameter(torch.FloatTensor(augmented_hidden_size))
        self.h_embedding = nn.Linear(2 * augmented_hidden_size, augmented_hidden_size, bias=False)
        self.BN = nn.BatchNorm1d(augmented_hidden_size)
        self.multi_head_embedding =  nn.Linear(self.n_multi_head * augmented_hidden_size, augmented_hidden_size, bias=False)
        self._initialize_weights(params["init_min"], params["init_max"])
        self.use_logit_clipping = params["use_logit_clipping"]
        self.C = params["C"]
        self.T = params["T"]
        self.n_glimpse = params["n_glimpse"]
        self.job_selecter = Categorical()





    def get_jssp_instance(self, instance): # 훈련해야할 instance를 에이전트가 참조(등록)하는 코드
        self.instance = instance
        self.mask1_temp = [instance.mask1 for instance in self.instance]
        self.mask2_temp = [instance.mask2 for instance in self.instance]

    def init_mask(self):
        dummy_instance = self.instance[0]
        shape0 = torch.tensor(dummy_instance.mask1).to(device).shape[0]
        shape1 = torch.tensor(dummy_instance.mask1).to(device).shape[1] # dummy_instance는 shape만 확인해 주기 위해 사용되는 instance
        mask1 = torch.zeros([len(self.instance), shape0, shape1]).to(device)
        mask2 = torch.zeros([len(self.instance), shape0, shape1]).to(device)
        for idx in range(len(self.instance)):                           # instance의 길이만큼 초기화
            instance = self.instance[idx]
            for i in range(len(instance.mask1)):                        # mask1(operation availability)에 대해서, 모든 Job의 첫번째 operation의 availability를 okay로 설정
                instance.mask1[i][0] = 1
            mask1[idx] = torch.tensor(instance.mask1).to(device)        # 현재 순서에 해당되는 batch data의 mask를 변경해 준다.
            mask2[idx] = torch.tensor(instance.mask2).to(device)
        return mask1, mask2

    def update_mask(self, job_selections):
        dummy_instance = self.instance[0]
        shape0 = torch.tensor(dummy_instance.mask1).to(device).shape[0]
        shape1 = torch.tensor(dummy_instance.mask1).to(device).shape[1]
        mask1 = torch.zeros([len(self.instance), shape0, shape1]).to(device)
        mask2 = torch.zeros([len(self.instance), shape0, shape1]).to(device)
        for idx in range(len(self.instance)):
            instance = self.instance[idx]
            job_selection = job_selections[idx]
            if 1 not in instance.mask1[job_selection]:
                pass
            else:
                index = instance.mask1[job_selection].index(1)
                instance.mask1[job_selection][index] = 0
                if index + 1 < len(instance.mask1[job_selection]):
                    instance.mask1[job_selection][index + 1] = 1

            if 0 not in instance.mask2[job_selection]:
                instance.mask2[job_selection][0] = 0
            else:
                if 1 in instance.mask2[job_selection]:
                    index = instance.mask2[job_selection].index(1)
                    instance.mask2[job_selection][index] = 0

            mask1[idx] = torch.tensor(instance.mask1).to(device)
            mask2[idx] = torch.tensor(instance.mask2).to(device)
       # empty=list()
        #for idx in range(len(self.instance)):
       #     empty.append(self.instance[idx].mask1.index(1))
        #print(empty)

        return mask1, mask2


    def _initialize_weights(self, init_min=-0.5, init_max=0.5):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def encoder(self, node_features, heterogeneous_edges):
        batch = node_features.shape[0]
        operation_num = node_features.shape[1]-2
        node_num = node_features.shape[1]
        node_reshaped_features = node_features.reshape(batch * node_num, -1)
        node_embedding = self.Embedding(node_reshaped_features)
        node_embedding = node_embedding.reshape(batch, node_num, -1)

        if self.k_hop == 1:
            enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
        if self.k_hop == 2:
            enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
            enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True, final = True)
        # if cfg.k_hop == 3:
        #     enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
        #     enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding2(heterogeneous_edges, enc_h, mini_batch=True, final = True)
        # if cfg.k_hop == 4:
        #     enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
        #     enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding2(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding3(heterogeneous_edges, enc_h, mini_batch=True, final = True)
        # if cfg.k_hop == 5:
        #     enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
        #     enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding2(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding3(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding4(heterogeneous_edges, enc_h, mini_batch=True, final = True)
        # if cfg.k_hop == 6:
        #     enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
        #     enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding2(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding3(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding4(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding5(heterogeneous_edges, enc_h, mini_batch=True, final = True)
        #
        # if cfg.k_hop == 7:
        #     enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
        #     enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding2(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding3(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding4(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding5(heterogeneous_edges, enc_h, mini_batch=True)
        #     enc_h = self.GraphEmbedding6(heterogeneous_edges, enc_h, mini_batch=True, final = True)

        embed = enc_h.size(2)
        h = enc_h.mean(dim = 1).unsqueeze(0) # 모든 node embedding에 대해서(element wise) 평균을 낸다.
        enc_h = enc_h[:, :-2]                 # dummy node(source, sink)는 제외한다.
        return enc_h, h, embed, batch, operation_num

    def get_critical_check(self, scheduler, mask):
        available_operations = mask
        avail_nodes = np.array(available_operations)
        avail_nodes_indices = np.where(avail_nodes == 1)[0].tolist() # 현재 시점에 가능한 operation들의 모임이다.
        scheduler.check_avail_ops(avail_nodes_indices)


    def branch_and_cut_masking(self, scheduler, mask, i, upperbound):
        available_operations = mask
        copied_mask = deepcopy(mask)
        copied_mask2 = deepcopy(mask)
        avail_nodes = np.array(available_operations)
        avail_nodes_indices = np.where(avail_nodes == 1)[0].tolist() # 현재 시점에 가능한 operation들의 모임이다.
        critical_path_list, critical_path_ij_list = scheduler.get_critical_path()
        if np.max(critical_path_list)>0:
            copied_mask[avail_nodes_indices] = critical_path_list
            copied_mask = copied_mask/np.max(critical_path_list)

        if np.max(critical_path_ij_list) > 0:
            copied_mask2[avail_nodes_indices] = critical_path_ij_list
            copied_mask2 = copied_mask2 / np.max(critical_path_ij_list)
        return mask, copied_mask, copied_mask2

####
    def forward(self, x, device, scheduler_list, num_job, num_machine, upperbound= None, old_sequence = None):
        node_features, heterogeneous_edges = x
        node_features = torch.tensor(node_features).to(device).float()
        pi_list, log_ps = [], []

        log_probabilities = list()


        sample_space = [[j for i in range(num_machine)] for j in range(num_job)]
        sample_space = torch.tensor(sample_space).view(-1)


        h_bar, h_emb, embed, batch, num_operations = self.encoder(node_features, heterogeneous_edges)

        """
        이 위에 까지가 Encoder
        이 아래 부터는 Decoder
    
        """

        h_pi_t_minus_one = self.v_1.unsqueeze(0).repeat(batch, 1).unsqueeze(0).to(device) # 이녀석이 s.o.s(start of singal)에 해당
        mask1_debug, mask2_debug = self.init_mask()
        batch_size = h_pi_t_minus_one.shape[1]

        if old_sequence != None:
            old_sequence = torch.tensor(old_sequence).long().to(device)
        next_operation_indices = list()
        for i in range(num_operations):
            est_placeholder = mask1_debug.clone().to(device)
            fin_placeholder = mask2_debug.clone().to(device)
            mask1_debug = mask1_debug.reshape(batch_size, -1)
            mask2_debug = mask2_debug.reshape(batch_size, -1)
            empty_zero = torch.zeros(batch_size, num_operations).to(device)
            empty_zero2 = torch.zeros(batch_size, num_operations).to(device)
            if i == 0:
                """
                Earliest Start Time (est_placeholder)
                Earliest Finish Time (fin_placeholder) 확인하는 로직
                i == 0일 때는 아직 선택된 operation이 없으므로,
                adaptive_run에 선택된 변수(i)에 대한 정보가 없음
                
                """
                for nb in range(batch_size):
                    scheduler_list[nb].adaptive_run(est_placeholder[nb], fin_placeholder[nb])

                    #self.get_critical_check(scheduler_list[nb],  mask1_debug[nb,:].cpu().numpy())
                    if (self.params['third_feature'] == "first_and_second") or\
                            (self.params['third_feature'] == "second_only"):
                        ub = upperbound[nb]
                        """
                        Branch and Cut 로직에 따라 masking을 수행함
                        모두 다 masking 처리할 수도 있으므로, 모두다 masking 할 경우에는 mask로 복원 (if 1 not in mask)

                        """
                        mask, critical_path, critical_path2 = self.branch_and_cut_masking(scheduler_list[nb], mask1_debug[nb,:].cpu().numpy(), i, upperbound = ub) # 안중요

                        empty_zero[nb, :] = torch.tensor(critical_path).to(device)# 안중요
                        empty_zero2[nb, :] = torch.tensor(critical_path2).to(device)  # 안중요
                        # if 1 not in mask:
                        #     pass
                        # else:
                        #     mask1_debug[nb, :] = torch.tensor(mask).to(device)

            else:
                """
                Earliest Start Time (est_placeholder)
                Earliest Finish Time (fin_placeholder) 확인하는 로직
                i == 0일 때는 아직 선택된 operation이 없으므로,
                adaptive_run에 선택된 변수(i)에 대한 정보는 이전에 선택된 index(next_operation_index)에서 추출

                """
                for nb in range(batch_size):
                    k = next_operation_index[nb].item()
                    scheduler_list[nb].add_selected_operation(k) # 그림으로 설명 예정# 안중요
                    next_b = next_job[nb].item()
                    scheduler_list[nb].adaptive_run(est_placeholder[nb], fin_placeholder[nb], i = next_b) # next_b는 이전 스텝에서 선택된 Job이고, Adaptive Run이라는 것은 선택된 Job에 따라 update한 다음에 EST, EFIN을 구하라는 의미
                    if (self.params['third_feature'] == "first_and_second") or\
                        (self.params['third_feature'] == "second_only"):
                        ub = upperbound[nb]
                        mask, critical_path, critical_path2 = self.branch_and_cut_masking(scheduler_list[nb], mask1_debug[nb,:].cpu().numpy(), i, upperbound = ub)
                        empty_zero[nb, :]  = torch.tensor(critical_path).to(device)
                        empty_zero2[nb, :] = torch.tensor(critical_path2).to(device)  # 안중요
                        """
                        Branch and Cut 로직에 따라 masking을 수행함
                        모두 다 masking 처리할 수도 있으므로, 모두다 masking할 경우에는 mask로 복원 (if 1 not in mask)
                        """
                        # if 1 not in mask:pass
                        # else:


            est_placeholder = est_placeholder.reshape(batch_size, -1).unsqueeze(2)
            fin_placeholder = fin_placeholder.reshape(batch_size, -1).unsqueeze(2)
            empty_zero = empty_zero.unsqueeze(2)
            empty_zero2 = empty_zero2.unsqueeze(2)
            if self.params['third_feature'] == "first_and_second":
                if self.params['ex_embedding'] == True:
                    r_temp = torch.concat([est_placeholder, fin_placeholder,empty_zero, empty_zero2], dim=2)  # extended node embedding을 만드는 부분(z_t_i에 해당)
                    r_temp = r_temp.reshape(batch_size*num_operations, -1)
                    ex_embedding = self.ex_embedding(r_temp)
                    ex_embedding = ex_embedding.reshape(batch_size, num_operations, -1)
                    ref = torch.concat([h_bar, ex_embedding],dim = 2)
                else:
                    ref = torch.concat([h_bar, est_placeholder, fin_placeholder, empty_zero, empty_zero2],dim = 2) # extended node embedding을 만드는 부분(z_t_i에 해당)
            if self.params['third_feature'] == "first_only":
                if self.params['ex_embedding'] == True:
                    r_temp = torch.concat([est_placeholder, fin_placeholder], dim=2)  # extended node embedding을 만드는 부분(z_t_i에 해당)
                    r_temp = r_temp.reshape(batch_size*num_operations, -1)
                    ex_embedding = self.ex_embedding(r_temp)
                    ex_embedding = ex_embedding.reshape(batch_size, num_operations, -1)
                    ref = torch.concat([h_bar, ex_embedding],dim = 2)
                else:
                    ref = torch.concat([h_bar, est_placeholder, fin_placeholder],dim=2)  # extended node embedding을 만드는 부분(z_t_i에 해당)
            if self.params['third_feature'] == "second_only":
                if self.params['ex_embedding'] == True:
                    r_temp = torch.concat([empty_zero, empty_zero2], dim=2)  # extended node embedding을 만드는 부분(z_t_i에 해당)
                    r_temp = r_temp.reshape(batch_size*num_operations, -1)
                    ex_embedding = self.ex_embedding(r_temp)
                    ex_embedding = ex_embedding.reshape(batch_size, num_operations, -1)
                    ref = torch.concat([h_bar, ex_embedding],dim = 2)
                else:
                    ref = torch.concat([h_bar, empty_zero, empty_zero2],dim=2)  # extended node embedding을 만드는 부분(z_t_i에 해당)


            h_c = self.decoder(h_emb, h_pi_t_minus_one) # decoding 만드는 부분
            query = h_c.squeeze(0)
            """
            Query를 만들때에는 이전 단계의 query와 extended node embedding을 가지고 만든다

            """
            query = self.glimpse(query, ref, mask2_debug)  # 보는 부분 /  multi-head attention 부분 (mask2는 보는 masking)
            logits = self.pointer(query, ref, mask1_debug) # 선택하는 부분 / logit 구하는 부분 (#mask1은 선택하는 masking)
            log_p = torch.log_softmax(logits / self.T, dim=-1) # log_softmax로 구하는 부분
            if old_sequence == None:
                next_operation_index = self.job_selecter(log_p)
            else:

                next_operation_index = old_sequence[i, :]

            log_probabilities.append(log_p.gather(1, next_operation_index.unsqueeze(1)))
            sample_space = sample_space.to(device)
            next_job = sample_space[next_operation_index].to(device)
            mask1_debug, mask2_debug = self.update_mask(next_job.tolist()) # update masking을 수행해주는
            h_pi_t_minus_one = torch.gather(input=h_bar, dim=1, index=next_operation_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed)).squeeze(1).unsqueeze(0)  # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
            next_operation_indices.append(next_operation_index.tolist())
            pi_list.append(next_job)



        pi = torch.stack(pi_list, dim=1)
        log_probabilities = torch.stack(log_probabilities, dim=1)
        ll = log_probabilities.sum(dim=1)    # 각 solution element의 log probability를 더하는 방식

        return pi, ll, next_operation_indices

    def glimpse(self, query, ref, mask0):
        """
        query는 decoder의 출력
        ref는   encoder의 출력
        """
        dk = self.params["n_hidden"]/self.n_multi_head
        for m in range(self.n_multi_head):
            u1 = self.W_q[m](query).unsqueeze(1)
            u2 = self.W_ref[m](ref.reshape(ref.shape[0]*ref.shape[1],-1))                             # u2: (batch, 128, block_num)
            u2 = u2.reshape(ref.shape[0], ref.shape[1],-1)
            u2 = u2.permute(0, 2, 1)
            u = torch.bmm(u1, u2)/dk**0.5
            v = ref@self.Vec[m]
            u = u.squeeze(1).masked_fill(mask0 == 0, -1e8)
            a = F.softmax(u, dim=1)
            if m == 0:
                g = torch.bmm(a.unsqueeze(1), v).squeeze(1)
            else:
                g += torch.bmm(a.unsqueeze(1), v).squeeze(1)
        query = g
        for m in range(self.n_multi_head):
            u1 = self.W_q3[m](query).unsqueeze(1)
            u2 = self.W_ref3[m](ref.reshape(ref.shape[0] * ref.shape[1], -1))  # u2: (batch, 128, block_num)
            u2 = u2.reshape(ref.shape[0], ref.shape[1], -1)
            u2 = u2.permute(0, 2, 1)
            u = torch.bmm(u1, u2) / dk ** 0.5
            v = ref @ self.Vec3[m]
            u = u.squeeze(1).masked_fill(mask0 == 0, -1e8)
            a = F.softmax(u, dim=1)
            if m == 0:
                g = torch.bmm(a.unsqueeze(1), v).squeeze(1)
            else:
                g += torch.bmm(a.unsqueeze(1), v).squeeze(1)


        query = g
        for m in range(self.n_multi_head):
            u1 = self.W_q4[m](query).unsqueeze(1)
            u2 = self.W_ref4[m](ref.reshape(ref.shape[0] * ref.shape[1], -1))  # u2: (batch, 128, block_num)
            u2 = u2.reshape(ref.shape[0], ref.shape[1], -1)
            u2 = u2.permute(0, 2, 1)
            u = torch.bmm(u1, u2) / dk ** 0.5
            v = ref @ self.Vec4[m]
            u = u.squeeze(1).masked_fill(mask0 == 0, -1e8)
            a = F.softmax(u, dim=1)
            if m == 0:
                g = torch.bmm(a.squeeze().unsqueeze(1), v).squeeze(1)
            else:
                g += torch.bmm(a.squeeze().unsqueeze(1), v).squeeze(1)

        return g

    def pointer(self, query, ref, mask, inf=1e8):
        if self.params["dot_product"] == False:
            u1 = self.W_q2(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, block_num)
            u2 = self.W_ref2(ref.permute(0, 2, 1))                         # u2: (batch, 128, block_num)
            V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
            u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
            if self.use_logit_clipping:
                u = self.C * torch.tanh(u)
            u = u.masked_fill(mask == 0, -1e8)
        else:
            dk = self.params["n_hidden"]
            u1 = self.W_q2(query).unsqueeze(1)
            u2 = self.W_ref2(ref.reshape(ref.shape[0] * ref.shape[1], -1))  # u2: (batch, 128, block_num)
            u2 = u2.reshape(ref.shape[0], ref.shape[1], -1)
            u2 = u2.permute(0, 2, 1)
            u = torch.bmm(u1, u2) / dk
            if self.use_logit_clipping:
                u = self.C * torch.tanh(u)
            u = u.squeeze(1).masked_fill(mask == 0, -1e8)
        return u

    def get_log_likelihood(self, _log_p, pi):
        log_p = torch.gather(input=_log_p, dim=2, index=pi)
        return torch.sum(log_p.squeeze(-1), dim = 2)

    def decoder(self, h_bar, h_t_minus_one):
        return torch.concat([h_bar, h_t_minus_one], dim =2)
