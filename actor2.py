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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


class PtrNet1(nn.Module):
    def __init__(self, params):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.gnn = params["gnn"]
        self.n_multi_head = params["n_multi_head"]
        if self.gnn == True:
            self.Embedding = nn.Linear(params["num_of_process"],  params["n_hidden"], bias=False).to(device)  # input_shape : num_of_process, output_shape : n_embedding
            if cfg.fully_connected == True:
                num_edge_cat = 4
            else:
                num_edge_cat = 3
            if cfg.gnn_type =='gcrl':
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
            else:
                num_edge_cat += 1
                self.GraphEmbedding = FastGTNs(num_edge_type=num_edge_cat,feature_size = params["n_hidden"],num_nodes = 102,num_FastGTN_layers = cfg.k_hop,hidden_size = params["n_hidden"],
                                    num_channels = self.n_multi_head,
                                    num_layers = cfg.k_hop,
                                    gtn_beta=0.05,
                                    teleport_probability = 'non-use')
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
            self.h_act = nn.ReLU()
            # self.mask = [[[0 for i in range(params['num_machine'])] for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.block_indices = []
            self.block_selecter = Categorical()  # {'greedy': Greedy(), 'sampling': Categorical()}.get(params["decode_type"], None)
            self.block_selecter_greedy = Greedy()
            self.last_block_index = 0
            self.params = params
            self.sample_space = [[j for i in range(params['num_machine'])] for j in range(params['num_jobs'])]
            self.sample_space = torch.tensor(self.sample_space).view(-1)

            self.mask_debug = [[[0 for i in range(params['num_machine'])] for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.mask_debug0 = [[[1 for i in range(params['num_machine'])] for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]

            self.job_count = [[0 for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.dummy_job_count = deepcopy(self.job_count)



        #print(self.job_count)
    def init_mask_job_count(self, count):
        self.count=count
        self.mask_debug = [[[0 for i in range(self.params['num_machine'])] for j in range(self.params['num_jobs'])] for _ in range(count)]
        self.mask_debug0 = [[[1 for i in range(self.params['num_machine'])] for j in range(self.params['num_jobs'])] for _ in
                            range(count)]
        self.job_count = [[0 for j in range(self.params['num_jobs'])] for _ in range(count)]

    def get_jssp_instance(self, instance):
        self.instance = instance
        self.mask1_temp = [instance.mask1 for instance in self.instance]
        self.mask2_temp = [instance.mask2 for instance in self.instance]

    def init_mask(self):
        dummy_instance = self.instance[0]
        shape0 = torch.tensor(dummy_instance.mask1).to(device).shape[0]
        shape1 = torch.tensor(dummy_instance.mask1).to(device).shape[1]
        mask1 = torch.zeros([len(self.instance), shape0, shape1]).to(device)
        mask2 = torch.zeros([len(self.instance), shape0, shape1]).to(device)
        for idx in range(len(self.instance)):
            instance = self.instance[idx]
            for i in range(len(instance.mask1)):
                instance.mask1[i][0] = 1

            mask1[idx] = torch.tensor(instance.mask1).to(device)




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
        block_num = node_features.shape[1]-2
        node_num = node_features.shape[1]
        node_reshaped_features = node_features.reshape(batch * node_num, -1)

        node_embedding = self.Embedding(node_reshaped_features)
        node_embedding = node_embedding.reshape(batch, node_num, -1)
        if cfg.gnn_type == 'gcrl':
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
            if cfg.k_hop == 5:
                enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
                enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True)
                enc_h = self.GraphEmbedding2(heterogeneous_edges, enc_h, mini_batch=True)
                enc_h = self.GraphEmbedding3(heterogeneous_edges, enc_h, mini_batch=True)
        else:

            batch = node_embedding.shape[0]
            enc_h = list()
            for b in range(batch):
                A = self.get_heterogeneous_adjacency_matrix(heterogeneous_edges[b][0], heterogeneous_edges[b][1], heterogeneous_edges[b][2],n_node_features=102)
                enc = self.GraphEmbedding(A, node_embedding[b])
                enc_h.append(enc)
            enc_h =torch.stack(enc_h, dim = 0)
        embed = enc_h.size(2)
        h = enc_h.mean(dim = 1).unsqueeze(0)
        enc_h = enc_h[:, :-2]
        return enc_h, h, embed, batch, block_num


    def forward(self, x, device, scheduler_list, y=None, greedy = False):
        node_features, heterogeneous_edges = x
        node_features = torch.tensor(node_features).to(device).float()
        pi_list, log_ps = [], []
        log_probabilities = list()
        enc_h, h_emb, embed, batch, num_operations = self.encoder(node_features, heterogeneous_edges)
        h_pi_t_minus_one = self.v_1.unsqueeze(0).repeat(batch, 1).unsqueeze(0).to(device)

        mask1_debug, mask2_debug = self.init_mask()
        ref_temp = enc_h
        batch_size = h_pi_t_minus_one.shape[1]
        for i in range(num_operations):

            mask1_debug = mask1_debug.reshape(batch_size, -1)
            mask2_debug = mask2_debug.reshape(batch_size, -1)


            job_count = torch.tensor(self.job_count)
            mask2 = torch.tensor(deepcopy(self.mask_debug), dtype=torch.float)
            for b in range(job_count.shape[0]):
                for k in range(job_count.shape[1]):
                    try: mask2[b, k, job_count[b, k]] = 1
                    except IndexError: pass


            est_placeholder = mask2.clone().to(device)
            fin_placeholder = mask2.clone().to(device)
            mask2 = mask2.view(self.count, -1).to(device)
            mask0 = torch.tensor(self.mask_debug0, dtype=torch.float).view(self.count, -1).to(device)
            #print(mask2.shape, mask0.shape)

            if i == 0:
                for nb in range(batch_size): # 하드코딩
                    scheduler_list[nb].adaptive_run(est_placeholder[nb], fin_placeholder[nb])
            else:
                for nb in range(batch_size):
                    next_b = next_block[nb].item()
                    scheduler_list[nb].adaptive_run(est_placeholder[nb], fin_placeholder[nb], i = next_b)
            est_placeholder = est_placeholder.reshape(batch, -1).unsqueeze(2)
            fin_placeholder = fin_placeholder.reshape(batch, -1).unsqueeze(2)
            ref = torch.concat([ref_temp, est_placeholder, fin_placeholder],dim = 2)
            h_c = self.decoder(h_emb, h_pi_t_minus_one)
            query = h_c.squeeze(0)
            query = self.glimpse(query, ref, mask2_debug)
            logits = self.pointer(query, ref, mask1_debug)
            log_p = torch.log_softmax(logits / self.T, dim=-1)
            if y == None:
                if greedy == False:
                    next_block_index = self.block_selecter(log_p)
                else:
                    next_block_index = self.block_selecter_greedy(log_p)
                log_probabilities.append(log_p.gather(1, next_block_index.unsqueeze(1)))
                self.block_indices.append(next_block_index)
                sample_space = self.sample_space.to(device)
                next_block = sample_space[next_block_index].to(device)

                mask1_debug, mask2_debug = self.update_mask(next_block.tolist())

                for b_prime in range(len(next_block.tolist())):
                    job = next_block[b_prime]
                    self.job_count[b_prime][job] += 1
                    mask_array = np.array(self.mask_debug0[b_prime][job])
                    idx = np.where(mask_array == 1)[0]
                    if idx.size > 0:
                        self.mask_debug0[b_prime][job][idx[0]] = 0
            else:
                log_p = log_p.squeeze(0)
                next_block_index = self.block_indices[i].long()
                log_probabilities.append(log_p.gather(1, next_block_index.unsqueeze(1)))
                sample_space = self.sample_space.to(device)
                next_block = sample_space[next_block_index].to(device)
                for b_prime in range(len(next_block.tolist())):
                    job = next_block[b_prime]
                    self.job_count[b_prime][job] += 1



            h_pi_t_minus_one = torch.gather(input=enc_h, dim=1, index=next_block_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed)).squeeze(1).unsqueeze(0)  # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
            pi_list.append(next_block)
        pi = torch.stack(pi_list, dim=1)
        log_probabilities = torch.stack(log_probabilities, dim=1)
        ll = log_probabilities.sum(dim=1)


        self.job_count = deepcopy(self.dummy_job_count)
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
