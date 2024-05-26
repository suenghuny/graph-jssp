import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from utils import _norm
import cfg

cfg = cfg.get_cfg()
from model import GCRN, NodeEmbedding
from model_fastgtn import FastGTNs
device = torch.device(cfg.device)

class Greedy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        #print("greedy", log_p.shape)
        return torch.argmax(log_p, dim=1).long()


class Categorical(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)


class Predictor(nn.Module):
    def __init_(self):
        super().__init__()
        self.fc1 = nn.linear()



class PtrNet1(nn.Module):
    def __init__(self, params):
        super().__init__()
        device = torch.device(cfg.device)
        self.n_multi_head = params["n_multi_head"]
        self.Embedding = nn.Linear(params["num_of_process"],  params["n_hidden"], bias=False).to(device)  # input_shape : num_of_process, output_shape : n_embedding

        num_edge_cat = 3
        self.GraphEmbedding = GCRN(feature_size =  params["n_hidden"],
                                   graph_embedding_size= params["graph_embedding_size"],
                                   embedding_size =  params["n_hidden"],layers =  params["layers"], num_edge_cat = num_edge_cat).to(device)
        self.GraphEmbedding1 = GCRN(feature_size =  params["n_hidden"],
                                   graph_embedding_size= params["graph_embedding_size"],
                                   embedding_size =  params["n_hidden"],layers =  params["layers"], num_edge_cat = num_edge_cat).to(device)
        self.GraphEmbedding2 = GCRN(feature_size= params["n_hidden"],
                                    graph_embedding_size=params["graph_embedding_size"],
                                    embedding_size=params["n_hidden"], layers=params["layers"],
                                    num_edge_cat=num_edge_cat).to(device)

        self.GraphEmbedding3 = GCRN(feature_size= params["n_hidden"],
                                    graph_embedding_size=params["graph_embedding_size"],
                                    embedding_size=params["n_hidden"], layers=params["layers"],
                                    num_edge_cat=num_edge_cat).to(device)
        self.GraphEmbedding4 = GCRN(feature_size= params["n_hidden"],
                                    graph_embedding_size=params["graph_embedding_size"],
                                    embedding_size=params["n_hidden"], layers=params["layers"],
                                    num_edge_cat=num_edge_cat).to(device)

        augmented_hidden_size = params["n_hidden"]
        self.Vec = [nn.Parameter(torch.FloatTensor(augmented_hidden_size+2, augmented_hidden_size)) for _ in range(self.n_multi_head)]
        self.Vec = nn.ParameterList(self.Vec)
        self.W_q = [nn.Linear(2*augmented_hidden_size, augmented_hidden_size, bias=False).to(device)  for _ in range(self.n_multi_head)]
        self.W_q_weights = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q])
        self.W_q_biases = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_q])
        self.W_ref =[nn.Linear(augmented_hidden_size+2,augmented_hidden_size, bias=False).to(device) for _ in range(self.n_multi_head)]
        self.W_ref_weights = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_ref])
        self.W_ref_biases = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_ref])
        self.Vec3 = [nn.Parameter(torch.FloatTensor(augmented_hidden_size+2, augmented_hidden_size)) for _ in range(self.n_multi_head)]
        self.Vec3 = nn.ParameterList(self.Vec3)
        self.W_q3 = [nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False).to(device)  for _ in range(self.n_multi_head)]
        self.W_q_weights3 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q3])
        self.W_q_biases3 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_q3])
        self.W_ref3 =[nn.Linear(augmented_hidden_size+2,augmented_hidden_size, bias=False).to(device) for _ in range(self.n_multi_head)]
        self.W_ref_weights3 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_ref3])
        self.W_ref_biases3 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_ref3])
        self.Vec4 = [nn.Parameter(torch.FloatTensor(augmented_hidden_size+2, augmented_hidden_size)) for _ in range(self.n_multi_head)]
        self.Vec4 = nn.ParameterList(self.Vec4)
        self.W_q4 = [nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False).to(device)  for _ in range(self.n_multi_head)]
        self.W_q_weights4 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q4])
        self.W_q_biases4 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_q4])
        self.W_ref4 =[nn.Linear(augmented_hidden_size+2,augmented_hidden_size, bias=False).to(device) for _ in range(self.n_multi_head)]
        self.W_ref_weights4 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_ref4])
        self.W_ref_biases4 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_ref4])
        self.Vec2 = nn.Parameter(torch.FloatTensor(augmented_hidden_size))
        self.W_q2 = nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False)
        self.W_ref2 = nn.Linear(augmented_hidden_size+2,augmented_hidden_size, bias=False)
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
        self.params = params


    def get_jssp_instance(self, instance):
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
                instance.mask1[job_selection][0] = 1
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

        if cfg.k_hop == 1:
            enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
        if cfg.k_hop == 2:
            enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
            enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True, final = True)
        if cfg.k_hop == 3:
            enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
            enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True)
            enc_h = self.GraphEmbedding2(heterogeneous_edges, enc_h, mini_batch=True, final = True)
        if cfg.k_hop == 4:
            enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
            enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True)
            enc_h = self.GraphEmbedding2(heterogeneous_edges, enc_h, mini_batch=True)
            enc_h = self.GraphEmbedding3(heterogeneous_edges, enc_h, mini_batch=True, final = True)

        embed = enc_h.size(2)
        h = enc_h.mean(dim = 1).unsqueeze(0) # 모든 node embedding에 대한 평균을 낸다.
        enc_h = enc_h[:, :-2]                # dummy node(source, sink)는 제외한다.
        return enc_h, h, embed, batch, operation_num

    def branch_and_cut_masking(self, scheduler, mask, i, upperbound):
        available_operations = mask
        avail_nodes = np.array(available_operations)
        avail_nodes_indices = np.where(avail_nodes == 1)[0].tolist() # 현재 시점에 가능한 operation들의 모임이다.
        critical_path_list = scheduler.get_critical_path()
        for i in range(len(avail_nodes_indices)):
            k = avail_nodes_indices[i]
            upperbound = upperbound
            if upperbound != None:
                if critical_path_list[i] >= upperbound: # 해당 operation을 선택했을 때, upperbound보다 크다는 건 해볼 가치가 없다고 볼 수 있음. 그래서 masking함
                    mask[k] = 0
        return mask




    def forward(self, x, device, scheduler_list, num_job, num_machine, upperbound= None):
        node_features, heterogeneous_edges = x
        node_features = torch.tensor(node_features).to(device).float()
        pi_list, log_ps = [], []
        log_probabilities = list()
        sample_space = [[j for i in range(num_machine)] for j in range(num_job)]
        sample_space = torch.tensor(sample_space).view(-1)

        h_bar, h_emb, embed, batch, num_operations = self.encoder(node_features, heterogeneous_edges)
        h_pi_t_minus_one = self.v_1.unsqueeze(0).repeat(batch, 1).unsqueeze(0).to(device)



        mask1_debug, mask2_debug = self.init_mask()
        batch_size = h_pi_t_minus_one.shape[1]

        for i in range(num_operations):
            est_placeholder = mask1_debug.clone().to(device)
            fin_placeholder = mask2_debug.clone().to(device)
            mask1_debug = mask1_debug.reshape(batch_size, -1)
            mask2_debug = mask2_debug.reshape(batch_size, -1)
            if i == 0:
                """
                Earliest Start Time (est_placeholder)
                Earliest Finish Time (fin_placeholder) 확인하는 로직
                i == 0일 때는 아직 선택된 operation이 없으므로,
                adaptive_run에 선택된 변수(i)에 대한 정보가 없음
                
                """
                for nb in range(batch_size):
                    scheduler_list[nb].adaptive_run(est_placeholder[nb], fin_placeholder[nb])
                    ub = upperbound[nb]


                    """
                    Branch and Cut 로직에 따라 masking을 수행함
                    모두 다 masking 처리할 수도 있으므로, 모두다 masking할 경우에는 mask로 복원 (if 1 not in mask)
                    """
                    mask = self.branch_and_cut_masking(scheduler_list[nb], mask1_debug[nb,:].cpu().numpy(), i, upperbound = ub)
                    if 1 not in mask:
                        pass
                    else:
                        mask1_debug[nb, :] = torch.tensor(mask).to(device)

            else:
                """
                Earliest Start Time (est_placeholder)
                Earliest Finish Time (fin_placeholder) 확인하는 로직
                i == 0일 때는 아직 선택된 operation이 없으므로,
                adaptive_run에 선택된 변수(i)에 대한 정보는 이전에 선택된 index(next_operation_index)에서 추출

                """
                for nb in range(batch_size):
                    k = next_operation_index[nb].item()
                    scheduler_list[nb].add_selected_operation(k) # 그림으로 설명 예정
                    next_b = next_job[nb].item()
                    scheduler_list[nb].adaptive_run(est_placeholder[nb], fin_placeholder[nb], i = next_b)
                    ub = upperbound[nb]
                    mask = self.branch_and_cut_masking(scheduler_list[nb], mask1_debug[nb,:].cpu().numpy(), i, upperbound = ub)
                    """
                    Branch and Cut 로직에 따라 masking을 수행함
                    모두 다 masking 처리할 수도 있으므로, 모두다 masking할 경우에는 mask로 복원 (if 1 not in mask)
                    """
                    if 1 not in mask:pass
                    else:

                        mask1_debug[nb, :] = torch.tensor(mask).to(device)

            est_placeholder = est_placeholder.reshape(batch_size, -1) * mask1_debug
            fin_placeholder = fin_placeholder.reshape(batch_size, -1) * mask1_debug
            est_placeholder = est_placeholder.reshape(batch_size, -1).unsqueeze(2)
            fin_placeholder = fin_placeholder.reshape(batch_size, -1).unsqueeze(2)
            ref = torch.concat([h_bar, est_placeholder, fin_placeholder],dim = 2) # additional information 만드는 부분

            h_c = self.decoder(h_emb, h_pi_t_minus_one) # decoding 만드는 부분
            query = h_c.squeeze(0)
            query = self.glimpse(query, ref, mask2_debug) # multi-head attention 부분
            logits = self.pointer(query, ref, mask1_debug) # logit 구하는 부분
            log_p = torch.log_softmax(logits / self.T, dim=-1) # log_softmax로 구하는 부분

            next_operation_index = self.job_selecter(log_p)
            log_probabilities.append(log_p.gather(1, next_operation_index.unsqueeze(1)))
            sample_space = sample_space.to(device)
            next_job = sample_space[next_operation_index].to(device)
            mask1_debug, mask2_debug = self.update_mask(next_job.tolist())
            h_pi_t_minus_one = torch.gather(input=h_bar, dim=1, index=next_operation_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed)).squeeze(1).unsqueeze(0)  # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
            pi_list.append(next_job)



        pi = torch.stack(pi_list, dim=1)

        print(pi.shape)
        log_probabilities = torch.stack(log_probabilities, dim=1)
        ll = log_probabilities.sum(dim=1)


        _ = 1
        return pi, ll, _

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
