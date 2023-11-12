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
        #print("cat", log_p.shape)
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



            #self.Vec = nn.Parameter(torch.cuda.FloatTensor(augmented_hidden_size))


            augmented_hidden_size = params["n_hidden"]
            #
            # self.Ws = nn.ParameterList(self.Ws)
            # [glorot(W) for W in self.Ws]

            self.Vec = [nn.Parameter(torch.cuda.FloatTensor(augmented_hidden_size, augmented_hidden_size)) for _ in range(self.n_multi_head)]
            self.Vec = nn.ParameterList(self.Vec)
            self.W_q = [nn.Linear(3*augmented_hidden_size, augmented_hidden_size, bias=False).to(device)  for _ in range(self.n_multi_head)]
            self.W_q_weights = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q])
            self.W_q_biases = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_q])
            self.W_ref =[nn.Linear(augmented_hidden_size,augmented_hidden_size, bias=False).to(device) for _ in range(self.n_multi_head)]
            self.W_ref_weights = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_ref])
            self.W_ref_biases = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_ref])

            self.Vec3 = [nn.Parameter(torch.cuda.FloatTensor(augmented_hidden_size, augmented_hidden_size)) for _ in range(self.n_multi_head)]
            self.Vec3 = nn.ParameterList(self.Vec3)
            self.W_q3 = [nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False).to(device)  for _ in range(self.n_multi_head)]
            self.W_q_weights3 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q3])
            self.W_q_biases3 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_q3])
            self.W_ref3 =[nn.Linear(augmented_hidden_size,augmented_hidden_size, bias=False).to(device) for _ in range(self.n_multi_head)]
            self.W_ref_weights3 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_ref3])
            self.W_ref_biases3 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_ref3])

            self.Vec4 = [nn.Parameter(torch.cuda.FloatTensor(augmented_hidden_size, augmented_hidden_size)) for _ in range(self.n_multi_head)]
            self.Vec4 = nn.ParameterList(self.Vec4)
            self.W_q4 = [nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False).to(device)  for _ in range(self.n_multi_head)]
            self.W_q_weights4 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q4])
            self.W_q_biases4 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_q4])
            self.W_ref4 =[nn.Linear(augmented_hidden_size,augmented_hidden_size, bias=False).to(device) for _ in range(self.n_multi_head)]
            self.W_ref_weights4 = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_ref4])
            self.W_ref_biases4 = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_ref4])



            self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(augmented_hidden_size))
            self.W_q2 = nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False)
            self.W_ref2 = nn.Linear(augmented_hidden_size,augmented_hidden_size, bias=False)
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
                enc_h = self.GraphEmbedding4(heterogeneous_edges, enc_h, mini_batch=True, final = True)
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


    def forward(self, x, device, y=None, greedy = False):
        node_features, heterogeneous_edges = x
        node_features = torch.tensor(node_features).to(device).float()
        batch = node_features.shape[0]
        block_num = node_features.shape[1] - 2

        pi_list, log_ps = [], []
        log_probabilities = list()


        h_pi_t_minus_one = self.v_1.unsqueeze(0).repeat(batch, 1).unsqueeze(0).to(device)
        h_pi_one = self.v_f.unsqueeze(0).repeat(batch, 1).unsqueeze(0).to(device)
        n_job = 10
        for i in range(block_num):
            copied_node_features = node_features.clone()
            job_count = torch.tensor(self.job_count)
            for b in range(batch):
                for j in range(n_job):
                    ops_id_init = j*n_job
                    ops_id = j*n_job+job_count[b][j]
                    ops_id_fin = j * n_job+ n_job
                    copied_node_features[b, ops_id_init:ops_id, -3:] = torch.tensor([1, 0, 0], dtype=torch.float)
                    copied_node_features[b, ops_id,    -3:] = torch.tensor([0, 1, 0], dtype = torch.float)
                    copied_node_features[b, ops_id+1:ops_id_fin, -3:] = torch.tensor([0, 0, 1], dtype=torch.float)
            enc_h, h, embed, batch, block_num = self.encoder(copied_node_features, heterogeneous_edges)
            ref = enc_h
            mask2 = torch.tensor(deepcopy(self.mask_debug), dtype=torch.float)
            for b in range(job_count.shape[0]):
                for k in range(job_count.shape[1]):
                    try:
                        mask2[b, k, job_count[b, k]] = 1
                    except IndexError as IE:
                        pass

            mask2 = mask2.view(self.count, -1).to(device)
            mask0 = torch.tensor(self.mask_debug0, dtype=torch.float).view(self.count, -1).to(device)
            h_c = self.decoder(h, h_pi_t_minus_one, h_pi_one)
            query = h_c.squeeze(0)
            query = self.glimpse(query, ref, mask0, mask2)
            logits = self.pointer(query, ref, mask2)
            log_p = torch.log_softmax(logits / self.T, dim=-1)
            if y == None:
                log_p = log_p.squeeze(0)
                if greedy == False:
                    next_block_index = self.block_selecter(log_p)
                else:
                    next_block_index = self.block_selecter_greedy(log_p)
                log_probabilities.append(log_p.gather(1, next_block_index.unsqueeze(1)))
                self.block_indices.append(next_block_index)
                sample_space = self.sample_space.to(device)
                next_block = sample_space[next_block_index].to(device)
                for b_prime in range(len(next_block.tolist())):
                    job = next_block[b_prime]
                    self.job_count[b_prime][job] += 1

                    mask_array = np.array(self.mask_debug0[b_prime][job])

                    # 첫 번째 1의 인덱스 찾기
                    idx = np.where(mask_array == 1)[0]

                    # 1이 존재하면 해당 위치를 0으로 변경
                    if idx.size > 0:
                        self.mask_debug0[b_prime][job][idx[0]] = 0


                    # for x in range(len(self.mask_debug0[b_prime][job])):
                    #     job_mask = self.mask_debug0[b_prime][job][x]
                    #     if job_mask == 1:
                    #         self.mask_debug0[b_prime][job][x] = 0
                    #         break
                        #print(self.mask_debug0[b_prime][job])
                    #print(self.mask_debug0[b_prime][job])
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
            if i == 0:
                h_pi_one = torch.gather(input=enc_h, dim=1, index=next_block_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed)).squeeze(1).unsqueeze(0)  # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
            #print(h_pi_one)
            pi_list.append(next_block)
        pi = torch.stack(pi_list, dim=1)
        if y == None:
            log_probabilities = torch.stack(log_probabilities, dim=1)
            ll = log_probabilities.sum(dim=1)
        else:
            log_probabilities = torch.stack(log_probabilities, dim=1)
            ll = log_probabilities.sum(dim=1)
        # print(self.block_indices[10])
        self.job_count = deepcopy(self.dummy_job_count)
        _ = 1
        return pi, ll, _

    def glimpse(self, query, ref, mask0, mask2, inf=1e8):
        """
        query는 decoder의 출력
        ref는   encoder의 출력
        """
        placeholder_for_g = list()

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
                g = torch.bmm(a.squeeze().unsqueeze(1), v).squeeze(1)
            else:
                g += torch.bmm(a.squeeze().unsqueeze(1), v).squeeze(1)
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
                g = torch.bmm(a.squeeze().unsqueeze(1), v).squeeze(1)
            else:
                g += torch.bmm(a.squeeze().unsqueeze(1), v).squeeze(1)

        # query = g
        # for m in range(self.n_multi_head):
        #     u1 = self.W_q4[m](query).unsqueeze(1)
        #     u2 = self.W_ref4[m](ref.reshape(ref.shape[0] * ref.shape[1], -1))  # u2: (batch, 128, block_num)
        #     u2 = u2.reshape(ref.shape[0], ref.shape[1], -1)
        #     u2 = u2.permute(0, 2, 1)
        #     u = torch.bmm(u1, u2) / dk ** 0.5
        #     v = ref @ self.Vec4[m]
        #     u = u.squeeze(1).masked_fill(mask2 == 0, -1e8)
        #     a = F.softmax(u, dim=1)
        #     if m == 0:
        #         g = torch.bmm(a.squeeze().unsqueeze(1), v).squeeze(1)
        #     else:
        #         g += torch.bmm(a.squeeze().unsqueeze(1), v).squeeze(1)

        # query = g
        # for m in range(self.n_multi_head):
        #     u1 = self.W_q4[m](query).unsqueeze(1)
        #     u2 = self.W_ref4[m](ref.reshape(ref.shape[0] * ref.shape[1], -1))  # u2: (batch, 128, block_num)
        #     u2 = u2.reshape(ref.shape[0], ref.shape[1], -1)
        #     u2 = u2.permute(0, 2, 1)
        #     u = torch.bmm(u1, u2) / dk ** 0.5
        #     v = ref @ self.Vec4[m]
        #     u = u.squeeze(1).masked_fill(mask == 0, -1e8)
        #     a = F.softmax(u, dim=1)
        #     # dddd
        #     if m == 0:
        #         g = torch.bmm(a.squeeze().unsqueeze(1), v).squeeze(1)
        #     else:
        #         g += torch.bmm(a.squeeze().unsqueeze(1), v).squeeze(1)
            #placeholder_for_g.append(g)
        #g = torch.concat(placeholder_for_g, dim = 1)
        #g = self.multi_head_embedding(g)

        return g

    def pointer(self, query, ref, mask, inf=1e8):

        if self.params["dot_product"] == False:
            u1 = self.W_q2(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, block_num)
            u2 = self.W_ref2(ref.permute(0, 2, 1))                         # u2: (batch, 128, block_num)
            V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
            u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
            if self.use_logit_clipping:
                u = self.C * torch.tanh(u)
            # V: (batch, 1, 128) * u1+u2: (batch, 128, block_num) => u: (batch, 1, block_num) => (batch, block_num)
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
            #print(u.shape, mask.shape)
            u = u.squeeze(1).masked_fill(mask == 0, -1e8)
        return u

    def get_log_likelihood(self, _log_p, pi):
        log_p = torch.gather(input=_log_p, dim=2, index=pi)
        return torch.sum(log_p.squeeze(-1), dim = 2)

    def decoder(self, h_bar, h_t_minus_one, h_one):
        #print(h_bar.shape, h_t_minus_one.shape, h_one.shape)
        return torch.concat([h_bar, h_t_minus_one, h_one], dim =2)

    def remove_done_operation(self, batched_hetero_edge_index, nodes_to_remove):
        batched_hetero_edge_index = [
            [[[1, 2, 3], [2, 3, 1]], [[3, 2, 1], [2, 3, 1]], [[4, 2, 1, 3], [2, 3, 1, 1]]],
            [[[2, 1, 3], [3, 3, 4]], [[3, 1, 1], [2, 3, 1]], [[5, 2, 1, 3], [1, 3, 4, 1]]]
        ]

        # 각 배치에서 제거할 노드 지정
        nodes_to_remove = [1, 2]  # 배치 1에서 노드 1, 배치 2에서 노드 2 제거

        # 결과를 저장할 리스트 초기화
        filtered_batched_hetero_edge_index = []

        # 각 배치를 순회하며 엣지 제거
        for batch_index, (hetero_edges, node_to_remove) in enumerate(zip(batched_hetero_edge_index, nodes_to_remove)):
            filtered_hetero_edges = []
            # 배치 내의 각 엣지 유형을 순회
            for edges in hetero_edges:
                # 각 엣지 유형 내의 엣지들을 순회
                filtered_edges = [[], []]
                for u, v in zip(*edges):
                    # 지정된 노드에 연결된 엣지가 아니라면 결과에 추가
                    if u != node_to_remove and v != node_to_remove:
                        filtered_edges[0].append(u)
                        filtered_edges[1].append(v)
                filtered_hetero_edges.append(filtered_edges)

            # 결과 리스트에 추가
            filtered_batched_hetero_edge_index.append(filtered_hetero_edges)

    def get_heterogeneous_adjacency_matrix(self, edge_index_1, edge_index_2, edge_index_3, n_node_features):
        A = []
        edge_index_1_transpose = deepcopy(edge_index_1)
        edge_index_1_transpose[1] = edge_index_1[0]
        edge_index_1_transpose[0] = edge_index_1[1]

        edge_index_2_transpose = deepcopy(edge_index_2)
        edge_index_2_transpose[1] = edge_index_2[0]
        edge_index_2_transpose[0] = edge_index_2[1]


        edge_index_3_transpose = deepcopy(edge_index_3)
        edge_index_3_transpose[1] = edge_index_3[0]
        edge_index_3_transpose[0] = edge_index_3[1]


        edges = [edge_index_1,
                 edge_index_2,
                 edge_index_3]

        for i, edge in enumerate(edges):
            edge = torch.tensor(edge, dtype=torch.long, device=device)
            value = torch.ones(edge.shape[1], dtype=torch.float, device=device)
            deg_inv_sqrt, deg_row, deg_col = _norm(edge.detach(),n_node_features,
                                                   value.detach())  # row의 의미는 차원이 1이상인 node들의 index를 의미함

            value = deg_inv_sqrt[
                        deg_row] * value  # degree_matrix의 inverse 중에서 row에 해당되는(즉, node의 차원이 1이상인) node들만 골라서 value_tmp를 곱한다
            A.append((edge, value))

        edge = torch.stack((torch.arange(0, n_node_features), torch.arange(0, n_node_features))).type(torch.LongTensor).to(device)
        value = torch.ones(n_node_features).type(torch.FloatTensor).to(device)
        A.append((edge, value))
        return A

    # def context_embedding(self, h_bar, pi_t_minus_one, h_one):
    #     h_context = self.get_h_context(self, h_bar, h_t_minus_one, h_one)
