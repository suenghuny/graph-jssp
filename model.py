import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
                self.linears['linear{}'.format(i)]= nn.Linear(last_layer, layer)
                self.linears['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.linears['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            else:
                self.linears['linear{}'.format(i)] = nn.Linear(last_layer, n_representation_obs)

        self.node_embedding = nn.Sequential(self.linears)
        #print(self.node_embedding)
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
        self.graph_embedding_size = graph_embedding_size
        self.embedding_size = embedding_size
        self.attention = attention
        self.Ws = []
        for i in range(num_edge_cat):
            self.Ws.append(nn.Parameter(torch.Tensor(feature_size, graph_embedding_size)))
        self.Ws = nn.ParameterList(self.Ws)
        [glorot(W) for W in self.Ws]

        self.Wv = []
        for i in range(num_edge_cat):
            #self.Wv.append(nn.Linear(feature_size, graph_embedding_size, bias = True).to(device))
            self.Wv.append(nn.Parameter(torch.Tensor(feature_size, graph_embedding_size)))
        self.Wv = nn.ParameterList(self.Wv)
        [glorot(W) for W in self.Wv]

        self.Wq = []
        for i in range(num_edge_cat):
            #self.Wq.append(nn.Linear(feature_size, graph_embedding_size, bias = True).to(device))
            self.Wq.append(nn.Parameter(torch.Tensor(feature_size, graph_embedding_size)))
        self.Wq = nn.ParameterList(self.Wq)
        [glorot(W) for W in self.Wq]

        self.embedding_layers = NodeEmbedding(graph_embedding_size*num_edge_cat, embedding_size, layers).to(device)

        self.m = [nn.ReLU() for _ in range(num_edge_cat)]
        # self.mv = [nn.Softplus() for _ in range(num_edge_cat)]
        # self.mq = [nn.Softplus() for _ in range(num_edge_cat)]

        self.a = [nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1))) for i in range(num_edge_cat)]
        [nn.init.xavier_uniform_(self.a[e].data, gain=1.414) for e in range(num_edge_cat)]


    #def forward(self, A, X, num_nodes=None, mini_batch=False):
    def _prepare_attentional_mechanism_input(self, Wq, Wv,A,k, mini_batch):
        Wh1 = Wq
        Wh2 = Wv
        # # print(Wh1.shape)
        # print(Wh1[0], Wh2.T[:, 0])
        # print(Wh1[0] @ Wh2.T[:, 0])
        # print(Wh1[1] @ Wh2.T[:, 1])
        # print(Wh1[-1] @ Wh2.T[:, -1])
        #print(Wh1.shape, Wh2.shape)
        #print(Wh1.shape, Wh2.shape)

        e = Wh1 @ Wh2.T
        # print(e)
        #print(e)
        #print(e.shape, A.shape)
        #print((e*A)[6])
        #print("ddd", self.m[k](e*A)[0], (e*A)[0])
        return self.m[k](e*A)

    def forward(self, A, X, mini_batch, layer = 0):

        if mini_batch == False:
            temp = list()
            for e in range(len(A)):

                E = A[e]
                num_nodes = X.shape[0]
                E = torch.sparse_coo_tensor(E, torch.ones(torch.tensor(E).shape[1]), (num_nodes, num_nodes)).to(device).to_dense()
                Wh = X @ self.Ws[e]
                Wq = self.mq[e](self.Wq[e](X))
                Wv = self.mv[e](self.Wv[e](X))


                a = self._prepare_attentional_mechanism_input(Wq, Wv, E, e, mini_batch = mini_batch)
                a = F.softmax(a*20, dim = 1)
                H = a*E@Wh

                temp.append(H)
            H = torch.cat(temp, dim = 1)
            #print(H.shape)
            H = self.embedding_layers(H)
            return H
        else:
            batch_size = X.shape[0]
            num_nodes = X.shape[1]
            #mat_a = [torch.zeros(self.num_edge_cat, num_nodes, num_nodes).to(device) for _ in range(batch_size)]
            empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)

            for b in range(batch_size):
                for e in range(self.num_edge_cat):
                    E = torch.sparse_coo_tensor(A[b][e],
                                            torch.ones(torch.tensor(torch.tensor(A[b][e]).shape[1])),
                                            (num_nodes, num_nodes)).to(device).to_dense()
                    # X_b = X[b].dou
                    #print(X[b])
                    H = self.m[e](E @ X[b] @ self.Ws[e])
                    # print(Wh.shape)
                    # Wq = self.mq[e](X[b] @ self.Wq[e])# self.mq[e](self.Wq[e](X[b]))
                    # Wv = self.mv[e](X[b] @ self.Wv[e])# self.mv[e](self.Wv[e](X[b]))
                    # # print(Wq[0])
                    # # print(Wq[4])
                    # # print("==================")
                    # a = self._prepare_attentional_mechanism_input(Wq, Wv,E, e, mini_batch=mini_batch)

                    # a = F.softmax(a*10, dim=1)
                    # H = a*E@Wh
                    empty[b, :, e, :].copy_(H)

            H = empty.reshape(batch_size, num_nodes, self.num_edge_cat*self.graph_embedding_size)
            H = H.reshape(batch_size*num_nodes, self.num_edge_cat*self.graph_embedding_size)
            H = self.embedding_layers(H)
            H = H.reshape(batch_size, num_nodes, self.embedding_size)
            return H