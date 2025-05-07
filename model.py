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
#
# class GCRN(nn.Module):
#     def __init__(self, feature_size, embedding_size, graph_embedding_size, layers, n_multi_head, num_edge_cat, alpha,  attention = False):
#         super(GCRN, self).__init__()
#         #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         self.num_edge_cat = num_edge_cat
#         self.feature_size =feature_size
#         self.graph_embedding_size = graph_embedding_size
#         self.embedding_size = embedding_size
#         self.attention = attention
#         self.Ws = []
#         self.n_multi_head = n_multi_head
#         #print(self.n_multi_head)
#         for i in range(num_edge_cat):
#             self.Ws.append(nn.Parameter(torch.Tensor(self.n_multi_head*feature_size, graph_embedding_size)))
#         self.Ws = nn.ParameterList(self.Ws)
#         [glorot(W) for W in self.Ws]
#         self.alpha = alpha
#
#
#         self.a1 = [nn.Parameter(torch.Tensor(size=(self.n_multi_head * graph_embedding_size, 1))) for _ in range(self.num_edge_cat)]
#         self.a1 = nn.ParameterList(self.a1)
#         [nn.init.xavier_uniform_(self.a1[k].data, gain=1.414) for k in range(self.num_edge_cat)]
#
#         self.a2 = [nn.Parameter(torch.Tensor(size=(self.n_multi_head * graph_embedding_size, 1))) for _ in range(self.num_edge_cat)]
#         self.a2 = nn.ParameterList(self.a2)
#         [nn.init.xavier_uniform_(self.a2[k].data, gain=1.414) for k in range(self.num_edge_cat)]
#
#         self.m = [nn.ReLU() for _ in range(num_edge_cat)]
#         self.leakyrelu = [nn.LeakyReLU() for _ in range(num_edge_cat)]
#         self.Embedding1 = nn.Linear(self.n_multi_head*graph_embedding_size*num_edge_cat, feature_size, bias = False)
#
#         self.Embedding1_mean = nn.Linear(graph_embedding_size * num_edge_cat, feature_size, bias = False)
#         self.Embedding2 = NodeEmbedding(feature_size, feature_size, layers = layers)
#         self.BN1 = nn.BatchNorm1d(feature_size)
#         self.BN2 = nn.BatchNorm1d(feature_size)
#
#     #def forward(self, A, X, num_nodes=None, mini_batch=False):
#     def _prepare_attentional_mechanism_input(self, Wq, Wv, e, m):
#         Wh1 = Wq
#         Wh1 = torch.matmul(Wh1, self.a1[e][m*self.graph_embedding_size:(m+1)*self.graph_embedding_size, :])
#         Wh2 = Wv
#         Wh2 = torch.matmul(Wh2, self.a2[e][m*self.graph_embedding_size:(m+1)*self.graph_embedding_size, :])
#         a = Wh1 + Wh2.T
#         return F.leaky_relu(a)
#
#     def forward(self, A, X, mini_batch, layer = 0, final = False):
#         batch_size = X.shape[0]
#         num_nodes = X.shape[1]
#         placeholder_for_multi_head = torch.zeros(batch_size, num_nodes, self.n_multi_head, self.num_edge_cat *self.graph_embedding_size).to(device)
#         if final == False:
#             pass
#         else:
#             empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)
#
#
#         for m in range(self.n_multi_head):
#             if final == False:
#                 empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)
#             else:
#                 pass
#
#             for b in range(batch_size):
#                 for e in range(self.num_edge_cat):
#                     E = torch.sparse_coo_tensor(A[b][e],torch.ones(torch.tensor(torch.tensor(A[b][e]).shape[1])),(num_nodes, num_nodes)).to(device).to_dense()
#                     Wh = X[b] @ self.Ws[e][m*self.feature_size:(m+1)*self.feature_size]
#                     a = self._prepare_attentional_mechanism_input(Wh, Wh, e, m)
#
#                     zero_vec = -9e15 * torch.ones_like(E)
#                     a = torch.where(E > 0, a, zero_vec)
#
#                     a = F.softmax(a, dim=1)
#                     H =  F.elu(torch.matmul(a, Wh))
#                     if final == False:
#                         empty[b, :, e, :] = H
#                     else:
#                         empty[b, :, e, :] += 1 / self.n_multi_head * H
#             if final == False:
#                 H = empty.reshape(batch_size, num_nodes, self.num_edge_cat*self.graph_embedding_size)
#                 placeholder_for_multi_head[:, :, m, :] = H
#             else:
#                 pass
# ###
#         if final == False:
#             H = placeholder_for_multi_head.reshape(batch_size, num_nodes, self.n_multi_head * self.num_edge_cat * self.graph_embedding_size)
#             H = H.reshape(batch_size*num_nodes, -1)
#             #print("1", H.shape, self.graph_embedding_size, self.feature_size)
#             H = self.Embedding1(H)
#             #print("2", H.shape)
#             X = X.reshape(batch_size*num_nodes, -1)
#             #print("3", X.shape)
#             H = self.BN1((1 - self.alpha)*H + self.alpha*X)
#             #print("4", X.shape)
#         else:
#             H = empty.reshape(batch_size, num_nodes, self.num_edge_cat * self.graph_embedding_size)
#             H = H.reshape(batch_size * num_nodes, -1)
#             H = self.Embedding1_mean(H)
#             X = X.reshape(batch_size * num_nodes, -1)
#             H = self.BN1((1 - self.alpha)*H + self.alpha*X)
#
#         Z = self.Embedding2(H)
#         #print("5", Z.shape)
#         Z = self.BN2(H + Z)
#         Z = Z.reshape(batch_size, num_nodes, -1)
#         return Z

# class GCRN(nn.Module):
#     def __init__(self, feature_size, embedding_size, graph_embedding_size, layers, n_multi_head, num_edge_cat, alpha,
#                  attention=False):
#         super(GCRN, self).__init__()
#         self.num_edge_cat = num_edge_cat
#         self.feature_size = feature_size
#         self.graph_embedding_size = graph_embedding_size
#         self.embedding_size = embedding_size
#         self.attention = attention
#         self.n_multi_head = n_multi_head
#
#         # 가중치 통합 텐서 (num_edge_cat, n_multi_head*feature_size, graph_embedding_size)
#         self.W = nn.Parameter(torch.Tensor(num_edge_cat, n_multi_head * feature_size, graph_embedding_size))
#         nn.init.xavier_uniform_(self.W, gain=1.414)
#
#         # 어텐션 가중치 통합 텐서
#         self.a1 = nn.Parameter(torch.Tensor(num_edge_cat, n_multi_head, graph_embedding_size, 1))
#         self.a2 = nn.Parameter(torch.Tensor(num_edge_cat, n_multi_head, graph_embedding_size, 1))
#         nn.init.xavier_uniform_(self.a1, gain=1.414)
#         nn.init.xavier_uniform_(self.a2, gain=1.414)
#
#         # 단일 활성화 함수
#         self.leakyrelu = nn.LeakyReLU()
#
#         # 임베딩 레이어
#         self.Embedding1 = nn.Linear(self.n_multi_head * graph_embedding_size * num_edge_cat, feature_size, bias=False)
#         self.Embedding1_mean = nn.Linear(graph_embedding_size * num_edge_cat, feature_size, bias=False)
#         self.Embedding2 = NodeEmbedding(feature_size, feature_size, layers=layers)
#
#         # 사전 계산된 상수
#         self.multi_head_dim = self.n_multi_head * self.num_edge_cat * self.graph_embedding_size
#
#     def _process_batch(self, A_batch, X_batch, m, e, num_nodes, device):
#         """단일 배치, 단일 어텐션 헤드, 단일 엣지 타입 처리 (마지막 남은 필수 루프)"""
#         batch_size = X_batch.shape[0]
#
#         # 가중치 슬라이스 (feature_size, graph_embedding_size)
#         W_slice = self.W[e, m * self.feature_size:(m + 1) * self.feature_size, :]
#
#         # 어텐션 가중치
#         a1_slice = self.a1[e, m, :, :]
#         a2_slice = self.a2[e, m, :, :]
#
#         # 배치 전체에 대한 행렬 곱 (B, N, F) @ (F, G) -> (B, N, G)
#         Wh = torch.bmm(X_batch, W_slice.expand(batch_size, self.feature_size, self.graph_embedding_size))
#
#         # 배치 병렬 어텐션 계산
#         # 1. (B, N, G) @ (G, 1) -> (B, N, 1)
#         Wh1 = torch.bmm(Wh, a1_slice.expand(batch_size, self.graph_embedding_size, 1))
#         Wh2 = torch.bmm(Wh, a2_slice.expand(batch_size, self.graph_embedding_size, 1))
#
#         # 2. 어텐션 스코어 계산 (B, N, N)
#         # Wh1: (B, N, 1), Wh2.transpose(1,2): (B, 1, N) -> broadcasting -> (B, N, N)
#         attention = Wh1 + Wh2.transpose(1, 2)
#         attention = self.leakyrelu(attention)
#
#         # 3. 인접 행렬로 마스킹
#         # A_batch: 희소 텐서 리스트를 밀집 텐서로 변환
#         # 이 부분은 PyTorch 텐서 연산만으로 최적화하기 어려운 부분
#
#         E_list = []
#         for b in range(batch_size):
#             adj_e = torch.tensor(A_batch[b][e]).to(device).long()
#             E = torch.sparse_coo_tensor(
#                 adj_e,
#                 torch.ones(adj_e.shape[1], device=device),
#                 (num_nodes, num_nodes)
#             ).to_dense()
#             E_list.append(E)
#
#         # 희소 행렬 스택
#         E_tensor = torch.stack(E_list)  # (batch_size, num_nodes, num_nodes)
#
#         # 전체 배치에 대한 마스킹 및 소프트맥스
#         masked_attn = torch.where(E_tensor > 0, attention, torch.tensor(-9e15, device=device))
#         attn_weights = F.softmax(masked_attn, dim=1)
#
#         # 배치 행렬 곱 한 번에 수행
#         results = F.elu(torch.bmm(attn_weights, Wh))
#
#         return results
#
#     def forward(self, A, X, mini_batch, layer=0, final=False):
#         batch_size, num_nodes, _ = X.shape
#         device = X.device
#
#         # 최종 결과를 위한 빈 텐서 - 초기화는 한 번만
#         if final:
#             final_result = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size,
#                                        device=device)
#         else:
#             multi_head_result = torch.zeros(batch_size, num_nodes, self.n_multi_head,
#                                             self.num_edge_cat * self.graph_embedding_size, device=device)
#
#         # 멀티헤드 & 엣지 타입 처리 - 병렬 계산으로 최대한 대체
#         # 공유 텐서 계산으로 루프 최소화
#         for m in range(self.n_multi_head):
#             # 각 멀티헤드별 결과를 위한 텐서
#             if not final:
#                 edge_results = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size,
#                                            device=device)
#
#             # 엣지 타입별 처리 - 병렬화 불가능한 부분
#             for e in range(self.num_edge_cat):
#                 # 배치에 대한 처리 병렬화
#                 H = self._process_batch(A, X, m, e, num_nodes, device)
#
#                 if final:
#                     final_result[:, :, e, :] += H / self.n_multi_head
#                 else:
#                     edge_results[:, :, e, :] = H
#
#             if not final:
#                 # reshape 대신 view 사용
#                 multi_head_result[:, :, m, :] = edge_results.view(batch_size, num_nodes,
#                                                                   self.num_edge_cat * self.graph_embedding_size)
#
#         # 최종 임베딩 계산
#         if not final:
#             # reshape를 view로 대체하여 속도 향상
#             H = multi_head_result.view(batch_size, num_nodes, self.multi_head_dim)
#             H = self.Embedding1(H.view(batch_size * num_nodes, -1))
#         else:
#             H = final_result.view(batch_size, num_nodes, self.num_edge_cat * self.graph_embedding_size)
#             H = self.Embedding1_mean(H.view(batch_size * num_nodes, -1))
#
#         # 마지막 임베딩 계산
#         Z = self.Embedding2(H)
#         return Z.view(batch_size, num_nodes, -1)

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

        edge_cat_tensor = torch.zeros(batch_size, num_nodes, num_nodes, self.num_edge_cat)
        for m in range(self.n_multi_head):
            if final == False:
                empty = torch.zeros(batch_size, num_nodes, self.num_edge_cat, self.graph_embedding_size).to(device)
            else:
                pass

            for b in range(batch_size):
                for e in range(self.num_edge_cat):

                    E = torch.sparse_coo_tensor(A[b][e],torch.ones(torch.tensor(torch.tensor(A[b][e]).shape[1])),(num_nodes, num_nodes)).to(device).to_dense()
                    edge_cat_tensor[b, :, :, e] = E
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
        Z = self.BN2(H + Z.clone())

        # 추가 수정 제안

        Z = Z.reshape(batch_size, num_nodes, -1)
        return Z, edge_cat_tensor