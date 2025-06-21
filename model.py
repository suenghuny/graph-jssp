import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
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
        self.feature_size = feature_size
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
            H = (1 - self.alpha)*H + self.alpha*X
            #print("4", X.shape)
        else:
            H = empty.reshape(batch_size, num_nodes, self.num_edge_cat * self.graph_embedding_size)
            H = H.reshape(batch_size * num_nodes, -1)
            H = self.Embedding1_mean(H)
            X = X.reshape(batch_size * num_nodes, -1)
            H = (1 - self.alpha)*H + self.alpha*X

        Z = self.Embedding2(H)
        Z = Z.reshape(batch_size, num_nodes, -1)
        return Z, edge_cat_tensor


class EnhancedPositionalEncoding(nn.Module):
    """
    액션 시퀀스에 위치 정보를 추가하고 MLP를 통과시켜 한 번 더 임베딩하는 향상된 Positional Encoding
    """

    def __init__(self, d_model, max_seq_length=100, mlp_hidden_dim=None, dropout=0.1):
        super(EnhancedPositionalEncoding, self).__init__()

        # MLP 히든 차원이 지정되지 않은 경우 기본값 설정
        if mlp_hidden_dim is None:
            mlp_hidden_dim = d_model * 2

        # 위치 인코딩 행렬 생성
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_seq_length, d_model]

        # 버퍼로 등록 (학습 파라미터는 아님)
        self.register_buffer('pe', pe)

        # MLP를 통한 추가 임베딩
        self.embedding_mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout)
        )

        # Layer Normalization (임베딩 전후의 특성을 보존하기 위해)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_length, d_model]
        Returns:
            위치 정보가 추가되고 MLP를 통과한 향상된 임베딩: [batch_size, seq_length, d_model]
        """
        # 위치 인코딩 적용
        x_with_pos = x + self.pe[:, :x.size(1), :]

        # 원본 형태 저장
        original_shape = x_with_pos.shape

        # 배치와 시퀀스 차원 합치기 (MLP 통과를 위해)
        x_reshaped = x_with_pos.view(-1, x_with_pos.size(-1))  # [batch_size * seq_length, d_model]

        # MLP 통과
        enhanced = self.embedding_mlp(x_reshaped)

        # 원래 형태로 복원
        enhanced = enhanced.view(original_shape)  # [batch_size, seq_length, d_model]

        # 잔차 연결 및 Layer Normalization
        enhanced = enhanced + x_with_pos

        return enhanced


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 모듈
    """

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // num_heads

        # Query, Key, Value 변환을 위한 선형 레이어
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        입력 텐서를 여러 개의 헤드로 분할
        Args:
            x: [batch_size, seq_length, d_model]
            batch_size: 배치 사이즈
        Returns:
            분할된 헤드: [batch_size, num_heads, seq_length, depth]
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, query_length, d_model]
            key: [batch_size, key_length, d_model]
            value: [batch_size, value_length, d_model]
            mask: 선택적 마스크 [batch_size, 1, 1, key_length]
        Returns:
            출력: [batch_size, query_length, d_model]
        """
        batch_size = query.size(0)

        # 선형 변환
        q = self.q_linear(query)  # [batch_size, query_length, d_model]
        k = self.k_linear(key)  # [batch_size, key_length, d_model]
        v = self.v_linear(value)  # [batch_size, value_length, d_model]

        # 헤드 분할
        q = self.split_heads(q, batch_size)  # [batch_size, num_heads, query_length, depth]
        k = self.split_heads(k, batch_size)  # [batch_size, num_heads, key_length, depth]
        v = self.split_heads(v, batch_size)  # [batch_size, num_heads, value_length, depth]

        # 스케일드 닷-프로덕트 어텐션 계산
        # q * k^T / sqrt(d_k)
        matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # [batch_size, num_heads, query_length, key_length]

        # 스케일링
        scaled_attention_logits = matmul_qk / math.sqrt(self.depth)

        # 마스킹 적용 (선택 사항)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax 적용
        attention_weights = F.softmax(scaled_attention_logits,
                                      dim=-1)  # [batch_size, num_heads, query_length, key_length]

        # 가중치와 value를 곱함
        output = torch.matmul(attention_weights, v)  # [batch_size, num_heads, query_length, depth]

        # 헤드 결합
        output = output.permute(0, 2, 1, 3).contiguous()  # [batch_size, query_length, num_heads, depth]
        output = output.view(batch_size, -1, self.d_model)  # [batch_size, query_length, d_model]

        # 최종 선형 변환
        output = self.out_linear(output)  # [batch_size, query_length, d_model]

        return output


class QValueAttentionModel(nn.Module):
    """
    강화학습에서 Q값을 출력하는 모델
    - latent variable z: 현재 상태(state)를 인코딩한 벡터, query 역할
    - action_sequence: 이전 행동 시퀀스나 가능한 행동들을 featurize한 시퀀스, key/value 역할
    """

    def __init__(self, state_dim, action_feature_dim, num_heads=4, mlp_hidden_dim=128,
                 pos_mlp_hidden_dim=None, state_projection_dim=None, dropout=0.1):
        super(QValueAttentionModel, self).__init__()

        self.state_dim = state_dim
        self.action_feature_dim = action_feature_dim

        # 상태 임베딩을 action_feature_dim과 맞추기 위한 projection 차원
        # 지정하지 않으면 action_feature_dim과 동일하게 설정
        self.state_projection_dim = state_projection_dim if state_projection_dim is not None else action_feature_dim

        # 상태(latent variable)를 action_feature_dim과 동일한 차원으로 확장하는 MLP
        self.state_projection = nn.Sequential(
            nn.Linear(state_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, self.state_projection_dim),
            nn.Dropout(dropout)
        )

        # 상태 확장 후 Layer Normalization
        self.state_norm = nn.LayerNorm(self.state_projection_dim)

        # 향상된 positional encoding (MLP 추가)
        self.pos_encoding = EnhancedPositionalEncoding(
            d_model=action_feature_dim,
            mlp_hidden_dim=pos_mlp_hidden_dim,
            dropout=dropout
        )

        # multi-head attention
        self.attention = MultiHeadAttention(self.state_projection_dim, num_heads)

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.state_projection_dim)

        # MLP - Q값 추정을 위한 네트워크
        self.q_mlp = nn.Sequential(
            nn.Linear(self.state_projection_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, 1)  # Q값(scalar) 출력
        )

    def forward(self, state_encoding, action_sequence, mask=None):
        """
        Args:
            state_encoding: [batch_size, state_dim] - 상태 인코딩 (latent z), query 역할
            action_sequence: [batch_size, seq_length, action_feature_dim] - 행동 시퀀스, key/value 역할
            mask: 선택적 마스크 [batch_size, 1, 1, seq_length] - 유효하지 않은 행동 마스킹
        Returns:
            Q값 출력: [batch_size, 1] - 각 상태-행동 쌍의 Q값
        """
        batch_size, seq_length, _ = action_sequence.size()

        # state_encoding을 MLP를 통해 확장
        # [batch_size, state_dim] -> [batch_size, state_projection_dim]
        projected_state = self.state_projection(state_encoding)
        #projected_state = self.state_norm(projected_state)

        # action sequence에 향상된 positional encoding 적용 (MLP를 통한 추가 임베딩 포함)
        action_sequence = self.pos_encoding(action_sequence)

        # 확장된 state를 attention query로 변환 (차원 확장)
        # [batch_size, state_projection_dim] -> [batch_size, 1, state_projection_dim]
        query = projected_state.unsqueeze(1)

        # multi-head attention 적용
        # query: [batch_size, 1, state_projection_dim] - 확장된 상태
        # key, value: [batch_size, seq_length, action_feature_dim] - 행동 시퀀스
        attn_output = self.attention(query, action_sequence, action_sequence,
                                     mask)  # [batch_size, 1, state_projection_dim]

        # add & norm (residual connection)
        attn_output = attn_output + query  # [batch_size, 1, state_projection_dim]

        # 차원 축소 [batch_size, 1, state_projection_dim] -> [batch_size, state_projection_dim]
        attn_output = attn_output.squeeze(1)

        # MLP를 통해 Q값(scalar) 출력
        q_value = self.q_mlp(attn_output)*1000  # [batch_size, 1]
        #q_value = F.softplus(q_value)

        return q_value