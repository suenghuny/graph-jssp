import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from model import GCRN, NodeEmbedding
from model_fastgtn import FastGTNs
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Greedy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.argmax(log_p, dim=1).long().squeeze(1)


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
            num_edge_cat = 4
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

            #self.Vec = nn.Parameter(torch.cuda.FloatTensor(augmented_hidden_size))


            augmented_hidden_size = params["n_hidden"]
            #
            # self.Ws = nn.ParameterList(self.Ws)
            # [glorot(W) for W in self.Ws]
            if torch.cuda.is_available():
                self.Vec = [nn.Parameter(torch.cuda.FloatTensor(augmented_hidden_size)) for _ in range(self.n_multi_head)]
                self.Vec = nn.ParameterList(self.Vec)
                self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(augmented_hidden_size))
            else:
                self.Vec = [nn.Parameter(torch.FloatTensor(augmented_hidden_size)) for _ in range(self.n_multi_head)]
                self.Vec = nn.ParameterList(self.Vec)
                self.Vec2 = nn.Parameter(torch.FloatTensor(augmented_hidden_size))


            self.W_q = [nn.Linear(3*augmented_hidden_size, augmented_hidden_size, bias=True).to(device)  for _ in range(self.n_multi_head)]
            self.W_q_weights = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_q])
            self.W_q_biases = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_q])
            self.W_ref =[nn.Conv1d(augmented_hidden_size,augmented_hidden_size, 1, 1).to(device) for _ in range(self.n_multi_head)]
            self.W_ref_weights = nn.ParameterList([nn.Parameter(q.weight) for q in self.W_ref])
            self.W_ref_biases = nn.ParameterList([nn.Parameter(q.bias) for q in self.W_ref])



            self.W_q2 = nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=True)

            self.W_ref2 = nn.Conv1d(augmented_hidden_size,augmented_hidden_size, 1, 1)
            self.dec_input = nn.Parameter(torch.FloatTensor(augmented_hidden_size))

            self.v_1 = nn.Parameter(torch.FloatTensor(augmented_hidden_size))
            self.v_f = nn.Parameter(torch.FloatTensor(augmented_hidden_size))
            self.h_embedding = nn.Linear(2 * augmented_hidden_size, augmented_hidden_size, bias=True)
            self.BN = nn.BatchNorm1d(augmented_hidden_size)
            self.multi_head_embedding =  nn.Linear(self.n_multi_head * augmented_hidden_size, augmented_hidden_size, bias=True)
            self.Decoder = nn.GRU(input_size=augmented_hidden_size,
                                  hidden_size=augmented_hidden_size, batch_first=True)


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
            self.job_count = [[0 for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.dummy_job_count = deepcopy(self.job_count)
        else:
            self.Embedding = nn.Linear(params["num_of_process"], params["n_hidden"], bias=False)  # input_shape : num_of_process, output_shape : n_embedding
            self.Encoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
            self.Decoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)


            if torch.cuda.is_available():
                self.Vec = nn.Parameter(torch.cuda.FloatTensor(params["n_hidden"]))
                self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(params["n_hidden"]))
            else:
                self.Vec = nn.Parameter(torch.FloatTensor(params["n_hidden"]))
                self.Vec2 = nn.Parameter(torch.FloatTensor(params["n_hidden"]))
            self.W_q = nn.Linear(3*params["n_hidden"], params["n_hidden"], bias=True)
            self.W_ref = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)

            self.W_q2 = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
            self.W_ref2 = nn.Conv1d(params["n_hidden"], params["n_hidden"], 1, 1)
            self.dec_input = nn.Parameter(torch.FloatTensor(params["n_embedding"]))
            self._initialize_weights(params["init_min"], params["init_max"])
            self.use_logit_clipping = params["use_logit_clipping"]
            self.C = params["C"]
            self.T = params["T"]
            self.n_glimpse = params["n_glimpse"]

            #self.mask = [[[0 for i in range(params['num_machine'])] for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.block_indices = []
            self.block_selecter =  Categorical() #{'greedy': Greedy(), 'sampling': Categorical()}.get(params["decode_type"], None)
            self.last_block_index = 0
            self.params = params
            self.sample_space = [[j for i in range(params['num_machine'])] for j in range(params['num_jobs'])]
            self.sample_space = torch.tensor(self.sample_space).view(-1)

            self.mask_debug = [[[0 for i in range(params['num_machine'])] for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.job_count = [[0 for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.dummy_job_count = deepcopy(self.job_count)


        #print(self.job_count)
    def init_mask_job_count(self, count):
        self.count=count
        self.mask_debug = [[[0 for i in range(self.params['num_machine'])] for j in range(self.params['num_jobs'])] for _ in range(count)]
        self.job_count = [[0 for j in range(self.params['num_jobs'])] for _ in range(count)]


    def _initialize_weights(self, init_min=-0.5, init_max=0.5):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x, device, y=None, greedy = False):

        if self.gnn == False:
            x = x.to(device)
            batch, block_num, _ = x.size()
            embed_enc_inputs = self.Embedding(x)
            embed = embed_enc_inputs.size(2)
            enc_h, h = self.Encoder(embed_enc_inputs, None)
        else:
            node_features, heterogeneous_edges = x
            node_features = torch.tensor(node_features).to(device).float()
            batch = node_features.shape[0]
            block_num = node_features.shape[1]-2
            node_num = node_features.shape[1]
            node_reshaped_features = node_features.reshape(batch * node_num, -1)
            node_embedding = self.Embedding(node_reshaped_features)
            node_embedding = node_embedding.reshape(batch, node_num, -1)
            enc_h = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
            enc_h = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True)
            enc_h = self.GraphEmbedding2(heterogeneous_edges, enc_h, mini_batch=True)
            embed = enc_h.size(2)
            #print(enc_h.shape)
            h = enc_h.mean(dim = 1).unsqueeze(0)
            enc_h = enc_h[:, :-2]


        ref = enc_h
        pi_list, log_ps = [], []
        dec_input = self.dec_input.unsqueeze(0).repeat(batch, 1).unsqueeze(1).to(device)
        log_probabilities = list()

        transformer = True
        if transformer == False:
            for i in range(block_num):
                job_count = torch.tensor(self.job_count)
                mask2 = torch.tensor(deepcopy(self.mask_debug), dtype = torch.float)
                for b in range(job_count.shape[0]):
                    for k in range(job_count.shape[1]):
                        try:
                            mask2[b, k, job_count[b, k]] =1
                        except IndexError as IE:
                            pass
                mask2= mask2.view(self.count, -1).to(device)
                _, h = self.Decoder(dec_input, h)
                #print(self.W_q.weight[0][10], self.W_q2.weight[0][15])

                query = h.squeeze(0)
                #print(query.shape, enc_h.shape)
                for j in range(self.n_glimpse):
                    query = self.glimpse(query, ref, mask2)
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
                else:
                    log_p = log_p.squeeze(0)
                    next_block_index = self.block_indices[i].long()
                    log_probabilities.append(log_p.gather(1, next_block_index.unsqueeze(1)))
                    #next_block = y[:, i].long()
                    sample_space = self.sample_space.to(device)
                    next_block = sample_space[next_block_index].to(device)
                    for b_prime in range(len(next_block.tolist())):
                        job = next_block[b_prime]
                        self.job_count[b_prime][job] += 1

                # if i == 10:
                #     print(log_p.gather(1, next_block_index.unsqueeze(1))[15])


                dec_input = torch.gather(input=enc_h, dim=1, index=next_block_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed)) # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
                pi_list.append(next_block)
            pi = torch.stack(pi_list, dim=1)
            if y == None:
                log_probabilities = torch.stack(log_probabilities, dim = 1)
                ll = log_probabilities.sum(dim=1)
            else:
                log_probabilities = torch.stack(log_probabilities, dim = 1)
                ll = log_probabilities.sum(dim=1)
            #print(self.block_indices[10])
            self.job_count = deepcopy(self.dummy_job_count)
            return pi, ll, _
        else:
            h_pi_t_minus_one = self.v_1.unsqueeze(0).repeat(batch, 1).unsqueeze(0).to(device)
            h_pi_one = self.v_f.unsqueeze(0).repeat(batch, 1).unsqueeze(0).to(device)

            for i in range(block_num):
                job_count = torch.tensor(self.job_count)
                mask2 = torch.tensor(deepcopy(self.mask_debug), dtype=torch.float)
                for b in range(job_count.shape[0]):
                    for k in range(job_count.shape[1]):
                        try:
                            mask2[b, k, job_count[b, k]] = 1
                        except IndexError as IE:
                            pass
                mask2 = mask2.view(self.count, -1).to(device)



                #print(h.shape, h_pi_t_minus_one.shape, h_pi_one.shape)
                h_c = self.decoder(h, h_pi_t_minus_one, h_pi_one)
                query = h_c.squeeze(0)
                query = self.glimpse(query, ref, mask2)
                logits = self.pointer(query, ref, mask2)
                log_p = torch.log_softmax(logits / self.T, dim=-1)
                if y == None:
                    log_p = log_p.squeeze(0)
                    next_block_index = self.block_selecter(log_p)
                    log_probabilities.append(log_p.gather(1, next_block_index.unsqueeze(1)))
                    self.block_indices.append(next_block_index)
                    sample_space = self.sample_space.to(device)
                    next_block = sample_space[next_block_index].to(device)
                    for b_prime in range(len(next_block.tolist())):
                        job = next_block[b_prime]
                        self.job_count[b_prime][job] += 1
                else:
                    log_p = log_p.squeeze(0)
                    next_block_index = self.block_indices[i].long()
                    log_probabilities.append(log_p.gather(1, next_block_index.unsqueeze(1)))
                    sample_space = self.sample_space.to(device)
                    next_block = sample_space[next_block_index].to(device)

                    for b_prime in range(len(next_block.tolist())):
                        job = next_block[b_prime]
                        self.job_count[b_prime][job] += 1

                # if i == 10:
                #     print(log_p.gather(1, next_block_index.unsqueeze(1))[15])

                h_pi_t_minus_one = torch.gather(input=enc_h, dim=1, index=next_block_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed)).squeeze(1).unsqueeze(0)  # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
                if i == 0:
                    h_pi_one = torch.gather(input=enc_h, dim=1, index=next_block_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed)).squeeze(1).unsqueeze(0)  # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
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

    def glimpse(self, query, ref, mask, inf=1e8):
        """
        query는 decoder의 출력
        ref는   encoder의 출력
        """
        placeholder_for_g = list()
        for m in range(self.n_multi_head):
            u1 = self.W_q[m](query).unsqueeze(-1).repeat(1, 1, ref.size(1))      # u1: (batch, 128, block_num)
            u2 = self.W_ref[m](ref.permute(0, 2, 1))                             # u2: (batch, 128, block_num)
            V = self.Vec[m].unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)  #
            u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
            u = u.masked_fill(mask == 0, -1e8)
            a = F.softmax(u, dim=1)
            g = torch.bmm(a.squeeze().unsqueeze(1), ref).squeeze(1)
            placeholder_for_g.append(g)
        g = torch.concat(placeholder_for_g, dim = 1)
        g = self.multi_head_embedding(g)
        g = self.BN(g)
        return g

    def pointer(self, query, ref, mask, inf=1e8):


        u1 = self.W_q2(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, block_num)
        u2 = self.W_ref2(ref.permute(0, 2, 1))                         # u2: (batch, 128, block_num)
        V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        if self.use_logit_clipping:
            u = self.C * torch.tanh(u)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, block_num) => u: (batch, 1, block_num) => (batch, block_num)
        u = u.masked_fill(mask == 0, -1e8)
        return u

    def get_log_likelihood(self, _log_p, pi):
        log_p = torch.gather(input=_log_p, dim=2, index=pi)
        return torch.sum(log_p.squeeze(-1), dim = 2)

    def decoder(self, h_bar, h_t_minus_one, h_one):
        #print(h_bar.shape, h_t_minus_one.shape, h_one.shape)
        return torch.concat([h_bar, h_t_minus_one, h_one], dim =2)

    # def context_embedding(self, h_bar, pi_t_minus_one, h_one):
    #     h_context = self.get_h_context(self, h_bar, h_t_minus_one, h_one)
