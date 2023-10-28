import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from model import GCRN


class Greedy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_p):
        return torch.argmax(log_p, dim=1).long()


class Categorical(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)


class PtrNet1(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.gnn = params["gnn"]
        if self.gnn == True:
            self.Embedding = nn.Linear(params["num_of_process"], params["n_embedding"],
                                       bias=False)  # input_shape : num_of_process, output_shape : n_embedding
            self.GraphEmbedding = GCRN(feature_size = params["num_of_process"],
                                       graph_embedding_size= params["graph_embedding_size"],
                                       embedding_size =  params["n_embedding"],layers =  params["layers"], num_edge_cat = 3)
            self.Decoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_embedding"], batch_first=True)
            if torch.cuda.is_available():
                self.Vec = nn.Parameter(torch.cuda.FloatTensor(params["n_embedding"]))
                self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(params["n_embedding"]))
            else:
                self.Vec = nn.Parameter(torch.FloatTensor(params["n_embedding"]))
                self.Vec2 = nn.Parameter(torch.FloatTensor(params["n_embedding"]))
            self.W_q = nn.Linear(params["n_embedding"], params["n_embedding"], bias=True)
            self.W_ref = nn.Conv1d(params["n_embedding"], params["n_embedding"], 1, 1)
            self.W_q2 = nn.Linear(params["n_embedding"], params["n_embedding"], bias=True)
            self.W_ref2 = nn.Conv1d(params["n_embedding"], params["n_embedding"], 1, 1)
            self.dec_input = nn.Parameter(torch.FloatTensor(params["n_embedding"]))

            self._initialize_weights(params["init_min"], params["init_max"])
            self.use_logit_clipping = params["use_logit_clipping"]
            self.C = params["C"]
            self.T = params["T"]
            self.n_glimpse = params["n_glimpse"]

            # self.mask = [[[0 for i in range(params['num_machine'])] for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.block_indices = []
            self.block_selecter = Categorical()  # {'greedy': Greedy(), 'sampling': Categorical()}.get(params["decode_type"], None)
            self.last_block_index = 0
            self.params = params
            self.sample_space = [[j for i in range(params['num_machine'])] for j in range(params['num_jobs'])]
            self.sample_space = torch.tensor(self.sample_space).view(-1)
            self.mask_debug = [[[-1e8 for i in range(params['num_machine'])] for j in range(params['num_jobs'])] for _
                               in range(params['batch_size'])]
            self.job_count = [[0 for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.dummy_job_count = deepcopy(self.job_count)
        else:
            self.Embedding = nn.Linear(params["num_of_process"], params["n_embedding"],
                                       bias=False)  # input_shape : num_of_process, output_shape : n_embedding
            self.Encoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
            self.Decoder = nn.LSTM(input_size=params["n_embedding"], hidden_size=params["n_hidden"], batch_first=True)
            if torch.cuda.is_available():
                self.Vec = nn.Parameter(torch.cuda.FloatTensor(params["n_hidden"]))
                self.Vec2 = nn.Parameter(torch.cuda.FloatTensor(params["n_hidden"]))
            else:
                self.Vec = nn.Parameter(torch.FloatTensor(params["n_hidden"]))
                self.Vec2 = nn.Parameter(torch.FloatTensor(params["n_hidden"]))
            self.W_q = nn.Linear(params["n_hidden"], params["n_hidden"], bias=True)
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
            self.mask_debug = [[[-1e8 for i in range(params['num_machine'])] for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.job_count = [[0 for j in range(params['num_jobs'])] for _ in range(params['batch_size'])]
            self.dummy_job_count = deepcopy(self.job_count)
        #print(self.job_count)

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, x, device, y=None):

        if self.gnn == False:
            x = x.to(device)
            batch, block_num, _ = x.size()
            embed_enc_inputs = self.Embedding(x)
            embed = embed_enc_inputs.size(2)
            enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
        else:
            node_features, heterogeneous_edges = x
            node_features = torch.tensor(node_features).to(device)
            embed_enc_inputs = self.Embedding(node_features)
            embed = embed_enc_inputs.size(2)
            batch = node_features.shape[0]
            block_num = node_features.shape[1]-2
            enc_h = self.GraphEmbedding(heterogeneous_edges,node_features,  mini_batch = True)
            c = enc_h[:, -1].unsqueeze(0).contiguous()
            h = enc_h[:, -2].unsqueeze(0).contiguous()
            enc_h = enc_h[:, :-2]

        ref = enc_h
        pi_list, log_ps = [], []

        dec_input = self.dec_input.unsqueeze(0).repeat(batch, 1).unsqueeze(1).to(device)

        log_probabilities = list()

        for i in range(block_num):
            job_count = torch.tensor(self.job_count)
            mask2 = torch.tensor(deepcopy(self.mask_debug))

            for b in range(job_count.shape[0]):
                for k in range(job_count.shape[1]):
                    try:
                        mask2[b, k, job_count[b, k]] = 0
                    except IndexError as IE:
                        pass
            mask2= mask2.view(self.params['batch_size'], -1).to(device)
            # ppp =[m for m in range(len(mask2[0].tolist()))  if mask2[0].tolist()[m] == 0]
            # print([self.sample_space.tolist()[p] for p in ppp])

            _, (h, c) = self.Decoder(dec_input, (h, c))
            query = h.squeeze(0)
            for j in range(self.n_glimpse):
                query = self.glimpse(query, ref, mask2)  # ref(enc_h)는 key이다.

            logits = self.pointer(query, ref, mask2)
            #print(logits.shape)
            log_p = torch.log_softmax(logits / self.T, dim=-1)
            #print(log_p[0][0].tolist().count(-1.0101e+08))
            if y == None:
                log_p = log_p.squeeze(0)
                next_block_index = self.block_selecter(log_p)
                log_probabilities.append(log_p.gather(1, next_block_index.unsqueeze(1)))
                self.block_indices.append(next_block_index)
                next_block = self.sample_space[next_block_index].to(device)

                for b_prime in range(len(next_block.tolist())):
                    job = next_block[b_prime]
                    self.job_count[b_prime][job] += 1
                #print(self.job_count)



                # for b in range(self.params['batch_size']):
                #     if 0 in self.mask[b][next_block[b]]:
                #         self.mask[b][next_block[b]][self.mask[b][next_block[b]].index(0)] = -1e8
            else:
                log_p = log_p.squeeze(0)
                next_block_index = self.block_indices[i].long()
                log_probabilities.append(log_p.gather(1, next_block_index.unsqueeze(1)))
                next_block = y[:, i].long()
                #print(y.shape, y.shape)
                #print("후",  next_block)
            #print(embed_enc_inputs.shape, next_block_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed).shape)

            dec_input = torch.gather(input=embed_enc_inputs, dim=1, index=next_block_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed)) # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
            log_p.detach().cpu()
            pi_list.append(next_block)
            log_ps.append(log_p)

        pi = torch.stack(pi_list, dim=1)
        ps = torch.stack(log_ps, dim=1)

        #print(torch.stack(log_ps, 1), torch.stack(self.block_indices, dim = 1).unsqueeze(0).to(device))
        if y == None:
            log_probabilities = torch.stack(log_probabilities, dim = 1)
            ll = log_probabilities.sum(dim=1)
        else:
            log_probabilities = torch.stack(log_probabilities, dim = 1)
            ll = log_probabilities.sum(dim=1)
        #self.mask = [[[0 for i in range(self.params['num_machine'])] for j in range(self.params['num_jobs'])] for _ in range(self.params['batch_size'])]
        #print(self.dummy_job_count)
        self.job_count = deepcopy(self.dummy_job_count)
        #print(self.job_count)
        return pi, ll, ps

    def glimpse(self, query, ref, mask, inf=1e8):
        """
        query는 decoder의 출력
        ref는   encoder의 출력
        """
        u1 = self.W_q(query).unsqueeze(-1).repeat(1, 1, ref.size(1))      # u1: (batch, 128, block_num)
        u2 = self.W_ref(ref.permute(0, 2, 1))                             # u2: (batch, 128, block_num)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)  #
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, block_num) => u: (batch, 1, block_num) => (batch, block_num)

        u = u + mask.unsqueeze(0)
        a = F.softmax(u, dim=1)
        #print(a.shape, a, a.sum(dim=1).shape)
        g = torch.bmm(a.squeeze().unsqueeze(1), ref).squeeze(1)

#        print(a, ref, g)

        return g

    def pointer(self, query, ref, mask, inf=1e8):
        u1 = self.W_q2(query).unsqueeze(-1).repeat(1, 1, ref.size(1))  # u1: (batch, 128, block_num)
        u2 = self.W_ref2(ref.permute(0, 2, 1))  # u2: (batch, 128, block_num)
        V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        if self.use_logit_clipping:
            u = self.C * torch.tanh(u)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, block_num) => u: (batch, 1, block_num) => (batch, block_num)
        u = u + mask.unsqueeze(0)

        return u

    def get_log_likelihood(self, _log_p, pi):
        log_p = torch.gather(input=_log_p, dim=2, index=pi)
        print(log_p.shape, torch.sum(log_p.squeeze(-1), dim = 2))
        return torch.sum(log_p.squeeze(-1), dim = 2)