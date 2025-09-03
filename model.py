import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from cfg import get_cfg
cfg = get_cfg()
from collections import OrderedDict
import sys

sys.path.append("..")  # 상위 폴더를 import할 수 있도록 경로 추가
device =torch.device(cfg.device)
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
    def __init__(self, feature_size, embedding_size, graph_embedding_size, layers, n_multi_head, num_edge_cat, alpha,  attention = False):
        super(GCRN, self).__init__()
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_edge_cat = num_edge_cat
        self.feature_size =feature_size
        self.graph_embedding_size = graph_embedding_size
        self.embedding_size = embedding_size
        self.attention = attention
        self.Ws = []
        self.n_multi_head = n_multi_head
        #print(self.n_multi_head)
        for i in range(num_edge_cat):
            self.Ws.append(nn.Parameter(torch.Tensor(self.n_multi_head*feature_size, graph_embedding_size)))
        self.Ws = nn.ParameterList(self.Ws)
        [glorot(W) for W in self.Ws]
        self.alpha = alpha


        self.a1 = [nn.Parameter(torch.Tensor(size=(self.n_multi_head * graph_embedding_size, 1))) for _ in range(self.num_edge_cat)]
        self.a1 = nn.ParameterList(self.a1)
        [nn.init.xavier_uniform_(self.a1[k].data, gain=1.414) for k in range(self.num_edge_cat)]

        self.a2 = [nn.Parameter(torch.Tensor(size=(self.n_multi_head * graph_embedding_size, 1))) for _ in range(self.num_edge_cat)]
        self.a2 = nn.ParameterList(self.a2)
        [nn.init.xavier_uniform_(self.a2[k].data, gain=1.414) for k in range(self.num_edge_cat)]

        self.m = [nn.ReLU() for _ in range(num_edge_cat)]
        self.leakyrelu = [nn.LeakyReLU() for _ in range(num_edge_cat)]
        self.Embedding1 = nn.Linear(self.n_multi_head*graph_embedding_size*num_edge_cat, feature_size, bias = False)

        self.Embedding1_mean = nn.Linear(graph_embedding_size * num_edge_cat, feature_size, bias = False)
        self.Embedding2 = NodeEmbedding(feature_size, feature_size, layers = layers)
        self.BN1 = nn.BatchNorm1d(feature_size)
        self.BN2 = nn.BatchNorm1d(feature_size)

    #def forward(self, A, X, num_nodes=None, mini_batch=False):
    def _prepare_attentional_mechanism_input(self, Wq, Wv, e, m):
        Wh1 = Wq
        Wh1 = torch.matmul(Wh1, self.a1[e][m*self.graph_embedding_size:(m+1)*self.graph_embedding_size, :])
        Wh2 = Wv
        Wh2 = torch.matmul(Wh2, self.a2[e][m*self.graph_embedding_size:(m+1)*self.graph_embedding_size, :])
        a = Wh1 + Wh2.T
        return F.leaky_relu(a)

    def forward(self, A, X, mini_batch, layer = 0, final = False):
        batch_size = X.shape[0]
        num_nodes = X.shape[1]
        placeholder_for_multi_head = torch.zeros(batch_size, num_nodes, self.n_multi_head, self.num_edge_cat *self.graph_embedding_size).to(device)
        if final == False:
            pass
        else:
            empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)


        for m in range(self.n_multi_head):
            if final == False:
                empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)
            else:
                pass

            for b in range(batch_size):
                for e in range(self.num_edge_cat):
                    E = torch.sparse_coo_tensor(A[b][e],torch.ones(torch.tensor(torch.tensor(A[b][e]).shape[1])),(num_nodes, num_nodes)).to(device).to_dense()
                    Wh = X[b] @ self.Ws[e][m*self.feature_size:(m+1)*self.feature_size]
                    a = self._prepare_attentional_mechanism_input(Wh, Wh, e, m)

                    zero_vec = -9e15 * torch.ones_like(E)
                    a = torch.where(E > 0, a, zero_vec)

                    a = F.softmax(a, dim=1)
                    H =  F.elu(torch.matmul(a, Wh))
                    if final == False:
                        empty[b, :, e, :] = H
                    else:
                        empty[b, :, e, :] += 1 / self.n_multi_head * H
            if final == False:
                H = empty.reshape(batch_size, num_nodes, self.num_edge_cat*self.graph_embedding_size)
                placeholder_for_multi_head[:, :, m, :] = H
            else:
                pass
###
        if final == False:
            H = placeholder_for_multi_head.reshape(batch_size, num_nodes, self.n_multi_head * self.num_edge_cat * self.graph_embedding_size)
            H = H.reshape(batch_size*num_nodes, -1)
            #print("1", H.shape, self.graph_embedding_size, self.feature_size)
            H = self.Embedding1(H)
            #print("2", H.shape)
            X = X.reshape(batch_size*num_nodes, -1)
            #print("3", X.shape)
            H = self.BN1((1 - self.alpha)*H + self.alpha*X)
            #print("4", X.shape)
        else:
            H = empty.reshape(batch_size, num_nodes, self.num_edge_cat * self.graph_embedding_size)
            H = H.reshape(batch_size * num_nodes, -1)
            H = self.Embedding1_mean(H)
            X = X.reshape(batch_size * num_nodes, -1)
            H = self.BN1((1 - self.alpha)*H + self.alpha*X)

        Z = self.Embedding2(H)
        #print("5", Z.shape)
        Z = self.BN2(H + Z)
        Z = Z.reshape(batch_size, num_nodes, -1)
        return Z



class GCN(nn.Module):
    def __init__(self, feature_size, embedding_size, graph_embedding_size, layers, n_multi_head, num_edge_cat, alpha,  attention = False):
        super(GCRN, self).__init__()
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_edge_cat = num_edge_cat
        self.feature_size =feature_size
        self.graph_embedding_size = graph_embedding_size
        self.embedding_size = embedding_size
        self.attention = attention
        self.Ws = []
        self.n_multi_head = n_multi_head
        #print(self.n_multi_head)
        for i in range(num_edge_cat):
            self.Ws.append(nn.Parameter(torch.Tensor(self.n_multi_head*feature_size, graph_embedding_size)))
        self.Ws = nn.ParameterList(self.Ws)
        [glorot(W) for W in self.Ws]
        self.alpha = alpha


        self.a1 = [nn.Parameter(torch.Tensor(size=(self.n_multi_head * graph_embedding_size, 1))) for _ in range(self.num_edge_cat)]
        self.a1 = nn.ParameterList(self.a1)
        [nn.init.xavier_uniform_(self.a1[k].data, gain=1.414) for k in range(self.num_edge_cat)]

        self.a2 = [nn.Parameter(torch.Tensor(size=(self.n_multi_head * graph_embedding_size, 1))) for _ in range(self.num_edge_cat)]
        self.a2 = nn.ParameterList(self.a2)
        [nn.init.xavier_uniform_(self.a2[k].data, gain=1.414) for k in range(self.num_edge_cat)]

        self.m = [nn.ReLU() for _ in range(num_edge_cat)]
        self.leakyrelu = [nn.LeakyReLU() for _ in range(num_edge_cat)]
        self.Embedding1 = nn.Linear(self.n_multi_head*graph_embedding_size*num_edge_cat, feature_size, bias = False)

        self.Embedding1_mean = nn.Linear(graph_embedding_size * num_edge_cat, feature_size, bias = False)
        self.Embedding2 = NodeEmbedding(feature_size, feature_size, layers = layers)
        self.BN1 = nn.BatchNorm1d(feature_size)
        self.BN2 = nn.BatchNorm1d(feature_size)

    #def forward(self, A, X, num_nodes=None, mini_batch=False):
    def _prepare_attentional_mechanism_input(self, Wq, Wv, e, m):
        Wh1 = Wq
        Wh1 = torch.matmul(Wh1, self.a1[e][m*self.graph_embedding_size:(m+1)*self.graph_embedding_size, :])
        Wh2 = Wv
        Wh2 = torch.matmul(Wh2, self.a2[e][m*self.graph_embedding_size:(m+1)*self.graph_embedding_size, :])
        a = Wh1 + Wh2.T
        return F.leaky_relu(a)

    def forward(self, A, X, mini_batch, layer = 0, final = False):
        batch_size = X.shape[0]
        num_nodes = X.shape[1]
        placeholder_for_multi_head = torch.zeros(batch_size, num_nodes, self.n_multi_head, self.num_edge_cat *self.graph_embedding_size).to(device)
        if final == False:
            pass
        else:
            empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)


        for m in range(self.n_multi_head):
            if final == False:
                empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)
            else:
                pass

            for b in range(batch_size):
                for e in range(self.num_edge_cat):
                    E = torch.sparse_coo_tensor(A[b][e],torch.ones(torch.tensor(torch.tensor(A[b][e]).shape[1])),(num_nodes, num_nodes)).to(device).to_dense()
                    Wh = X[b] @ self.Ws[e][m*self.feature_size:(m+1)*self.feature_size]
                    degree = torch.sum(E, dim=1, keepdim=True) + 1e-6  # avoid division by zero
                    D_inv_sqrt = torch.pow(degree, -0.5)

                    # Symmetric normalization: D^(-1/2) * A * D^(-1/2)
                    E_normalized = D_inv_sqrt * E * D_inv_sqrt.T

                    # Self-loop 추가 (optional, GCN에서 일반적)
                    E_normalized = E_normalized + torch.eye(num_nodes).to(device)

                    # GCN aggregation
                    H = F.relu(torch.matmul(E_normalized, Wh))
                    if final == False:
                        empty[b, :, e, :] = H
                    else:
                        empty[b, :, e, :] += 1 / self.n_multi_head * H
            if final == False:
                H = empty.reshape(batch_size, num_nodes, self.num_edge_cat*self.graph_embedding_size)
                placeholder_for_multi_head[:, :, m, :] = H
            else:
                pass
###
        if final == False:
            H = placeholder_for_multi_head.reshape(batch_size, num_nodes, self.n_multi_head * self.num_edge_cat * self.graph_embedding_size)
            H = H.reshape(batch_size*num_nodes, -1)
            #print("1", H.shape, self.graph_embedding_size, self.feature_size)
            H = self.Embedding1(H)
            #print("2", H.shape)
            X = X.reshape(batch_size*num_nodes, -1)
            #print("3", X.shape)
            H = self.BN1((1 - self.alpha)*H + self.alpha*X)
            #print("4", X.shape)
        else:
            H = empty.reshape(batch_size, num_nodes, self.num_edge_cat * self.graph_embedding_size)
            H = H.reshape(batch_size * num_nodes, -1)
            H = self.Embedding1_mean(H)
            X = X.reshape(batch_size * num_nodes, -1)
            H = self.BN1((1 - self.alpha)*H + self.alpha*X)

        Z = self.Embedding2(H)
        #print("5", Z.shape)
        Z = self.BN2(H + Z)
        Z = Z.reshape(batch_size, num_nodes, -1)
        return Z