import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from cfg import get_cfg
cfg = get_cfg()
from collections import OrderedDict
import sys

sys.path.append("..")  # 상위 폴더를 import할 수 있도록 경로 추가
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.device_count())
device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)

def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, n_representation_obs, layers = [20, 30 ,40]):
        super(NodeEmbedding, self).__init__()

        self.feature_size = feature_size
        self.linears = OrderedDict()
        last_layer = self.feature_size
        for i in range(len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.linears['linear{}'.format(i)]= nn.Linear(last_layer, layer, bias = False)
                self.linears['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.linears['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            else:
                self.linears['linear{}'.format(i)] = nn.Linear(last_layer, n_representation_obs, bias = False)
        self.node_embedding = nn.Sequential(self.linears)
        self.node_embedding.apply(weight_init_xavier_uniform)


    def forward(self, node_feature, missile=False):
        #print(node_feature.shape)
        node_representation = self.node_embedding(node_feature)
        return node_representation

class GCRN(nn.Module):
    def __init__(self, feature_size, embedding_size, graph_embedding_size, layers, num_edge_cat, attention = False):
        super(GCRN, self).__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_edge_cat = num_edge_cat
        self.feature_size =feature_size
        self.graph_embedding_size = graph_embedding_size
        self.embedding_size = embedding_size
        self.attention = attention
        self.Ws = []
        self.n_multi_head = cfg.n_multi_head
        for i in range(num_edge_cat):
            self.Ws.append(nn.Parameter(torch.Tensor(self.n_multi_head*feature_size, graph_embedding_size)))
        self.Ws = nn.ParameterList(self.Ws)
        [glorot(W) for W in self.Ws]

        self.Wv = []
        for i in range(num_edge_cat):
            self.Wv.append(nn.Parameter(torch.Tensor(self.n_multi_head*feature_size, 1)))
        self.Wv = nn.ParameterList(self.Wv)
        [glorot(W) for W in self.Wv]

        self.Wq = []
        for i in range(num_edge_cat):
            self.Wq.append(nn.Parameter(torch.Tensor(self.n_multi_head*feature_size, 1)))
        self.Wq = nn.ParameterList(self.Wq)
        [glorot(W) for W in self.Wq]
        #self.embedding_layers = NodeEmbedding(graph_embedding_size*num_edge_cat, embedding_size, layers).to(device)

        self.m = [nn.ReLU() for _ in range(num_edge_cat)]
        self.leakyrelu = [nn.LeakyReLU() for _ in range(num_edge_cat)]
        self.Embedding1 = nn.Linear(self.n_multi_head*graph_embedding_size*num_edge_cat, feature_size, bias = False)

        self.Embedding1_mean = nn.Linear(graph_embedding_size * num_edge_cat, feature_size, bias = False)
        self.Embedding2 = NodeEmbedding(feature_size, feature_size, layers = layers)
        self.BN1 = nn.BatchNorm1d(feature_size)
        self.BN2 = nn.BatchNorm1d(feature_size)

    #def forward(self, A, X, num_nodes=None, mini_batch=False):
    def _prepare_attentional_mechanism_input(self, Wq, Wv,A,k, mini_batch):
        Wh1 = Wq
        Wh2 = Wv
        e = Wh1 @ Wh2.T
        return e*A
    def forward(self, A, X, mini_batch, layer = 0, final = False):
        batch_size = X.shape[0]
        num_nodes = X.shape[1]

        placeholder_for_multi_head = []
        if final == False:pass
        else:
            empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)
        for m in range(self.n_multi_head):
            if final == False:
                empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)
            else:pass
            for b in range(batch_size):
                for e in range(self.num_edge_cat):
                    E = torch.sparse_coo_tensor(A[b][e],torch.ones(torch.tensor(torch.tensor(A[b][e]).shape[1])),(num_nodes, num_nodes)).to(device).to_dense()
                    Wh = X[b] @ self.Ws[e][m*self.feature_size:(m+1)*self.feature_size]
                    Wq = X[b] @ self.Wq[e][m*self.feature_size:(m+1)*self.feature_size]
                    Wv = X[b] @ self.Wv[e][m*self.feature_size:(m+1)*self.feature_size]
                    a = self._prepare_attentional_mechanism_input(Wq, Wv,E, e, mini_batch=mini_batch)
                    a = F.leaky_relu(a)
                    a = F.softmax(a, dim=1)
                    H = a@Wh
                    if final == False:
                        empty[b, :, e, :].copy_(H)
                    else:
                        empty[b, :, e, :] += 1 / self.n_multi_head * H
            if final == False:
                H = empty.reshape(batch_size, num_nodes, self.num_edge_cat*self.graph_embedding_size)
                placeholder_for_multi_head.append(H)
            else:
                pass
###
        if final == False:
            H = torch.concat(placeholder_for_multi_head, dim = 2)

            H = H.reshape(batch_size*num_nodes, -1)
            H = F.relu(self.Embedding1(H))
            #print(H.shape)
            X = X.reshape(batch_size*num_nodes, -1)
            H = self.BN1((1 - cfg.alpha)*H + cfg.alpha*X)
            #H = F.dropout(H, p = cfg.dropout)
        else:
            H = empty.reshape(batch_size, num_nodes, self.num_edge_cat * self.graph_embedding_size)
            H = H.reshape(batch_size * num_nodes, -1)
            H = F.relu(self.Embedding1_mean(H))
            X = X.reshape(batch_size * num_nodes, -1)
            H = self.BN1((1 - cfg.alpha)*H + cfg.alpha*X)
            #H = F.dropout(H, p = cfg.dropout)

        Z = self.Embedding2(H)
        Z = self.BN2(H + Z)
        #Z = F.dropout(Z)
        Z = Z.reshape(batch_size, num_nodes, -1)
        return Z