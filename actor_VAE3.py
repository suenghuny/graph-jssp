import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import numpy as np
import cfg
from copy import deepcopy
cfg = cfg.get_cfg()
from model import *
from latent import LatentModel
device = torch.device(cfg.device)



class Categorical(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)

class Categorical2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)

class ExEmbedding(nn.Module):
    def __init__(self, raw_feature_size, feature_size):
        super().__init__()
        self.fcn1 = nn.Linear(raw_feature_size, 84)
        self.fcn2 = nn.Linear(84, 64)
        self.fcn3 = nn.Linear(64, feature_size)
    def forward(self, x):
        x = F.elu(self.fcn1(x))
        x = F.elu(self.fcn2(x))
        x = self.fcn3(x)
        return x

class Critic(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.fcn1 = nn.Linear(z_size, 128)
        self.fcn2 = nn.Linear(128, 64)
        self.fcn3 = nn.Linear(64, 32)
        self.fcn4 = nn.Linear(32, 16)
        self.fcn5 = nn.Linear(16, 1)
    def forward(self, x):
        x = F.elu(self.fcn1(x))
        x = F.elu(self.fcn2(x))
        x = F.elu(self.fcn3(x))
        x = F.elu(self.fcn4(x))
        x = self.fcn5(x)
        return x


class PtrNet1(nn.Module):
    def __init__(self, params):
        super().__init__()
        device = torch.device(cfg.device)
        self.n_multi_head = params["n_multi_head"]
        self.params = params
        self.k_hop = params["k_hop"]

        num_edge_cat = 3
        z_dim = 144
        action_feature_dim = 4
        self.critic = QValueAttentionModel(
            state_dim=z_dim,
            action_feature_dim=action_feature_dim,
            num_heads=4,
            mlp_hidden_dim=128,
            pos_mlp_hidden_dim=action_feature_dim * 2,  # 추가된 파라미터
            dropout=0.1
        ).to(device)
        self.Latent = LatentModel(z_dim=z_dim, params = params).to(device)



        augmented_hidden_size = params["n_hidden"]

        self.ex_embedding = ExEmbedding(raw_feature_size=4, feature_size=params["ex_embedding_size"])

        # Vec 파라미터 리스트 생성 (문제 없음)
        self.Vec = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(augmented_hidden_size + params["ex_embedding_size"], augmented_hidden_size))
            for _ in range(self.n_multi_head)
        ])

        # 중복 파라미터 제거: W_q와 W_ref를 ModuleList로 변경
        self.W_q = nn.ModuleList([
            nn.Linear(augmented_hidden_size + z_dim, augmented_hidden_size + z_dim, bias=False).to(device)
            for _ in range(self.n_multi_head)
        ])

        self.W_ref = nn.ModuleList([
            nn.Linear(augmented_hidden_size + params["ex_embedding_size"], augmented_hidden_size + z_dim,
                      bias=False).to(device)
            for _ in range(self.n_multi_head)
        ])

        # 두 번째 어텐션 블록도 ModuleList로 변경
        self.Vec3 = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(augmented_hidden_size + params["ex_embedding_size"], augmented_hidden_size))
            for _ in range(self.n_multi_head)
        ])

        self.W_q3 = nn.ModuleList([
            nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False).to(device)
            for _ in range(self.n_multi_head)
        ])

        self.W_ref3 = nn.ModuleList([
            nn.Linear(augmented_hidden_size + params["ex_embedding_size"], augmented_hidden_size, bias=False).to(device)
            for _ in range(self.n_multi_head)
        ])

        # 세 번째 어텐션 블록도 ModuleList로 변경
        self.Vec4 = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(augmented_hidden_size + params["ex_embedding_size"], augmented_hidden_size))
            for _ in range(self.n_multi_head)
        ])

        self.W_q4 = nn.ModuleList([
            nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False).to(device)
            for _ in range(self.n_multi_head)
        ])

        self.W_ref4 = nn.ModuleList([
            nn.Linear(augmented_hidden_size + params["ex_embedding_size"], augmented_hidden_size, bias=False).to(device)
            for _ in range(self.n_multi_head)
        ])

        # 마지막 포인터 네트워크 관련 파라미터는 그대로 유지
        self.Vec2 = nn.Parameter(torch.FloatTensor(augmented_hidden_size))
        self.W_q2 = nn.Linear(augmented_hidden_size, augmented_hidden_size, bias=False)
        self.W_ref2 = nn.Linear(augmented_hidden_size + params["ex_embedding_size"], augmented_hidden_size, bias=False)
        self.v_1 = nn.Parameter(torch.FloatTensor(augmented_hidden_size))

        # 파라미터 목록 생성 방식도 변경
        # 모든 어텐션 관련 파라미터를 각 모듈에서 parameters() 메소드로 추출
        attention_params_1 = list(self.Vec) + [p for m in self.W_q for p in m.parameters()] + [p for m in self.W_ref for
                                                                                               p in m.parameters()]
        attention_params_2 = list(self.Vec3) + [p for m in self.W_q3 for p in m.parameters()] + [p for m in self.W_ref3
                                                                                                 for p in
                                                                                                 m.parameters()]
        attention_params_3 = list(self.Vec4) + [p for m in self.W_q4 for p in m.parameters()] + [p for m in self.W_ref4
                                                                                                 for p in
                                                                                                 m.parameters()]

        # 마지막 포인터 네트워크 관련 파라미터
        pointer_params = [self.Vec2, self.W_q2.weight]
        if self.W_q2.bias is not None:
            pointer_params.append(self.W_q2.bias)
        pointer_params.append(self.W_ref2.weight)
        if self.W_ref2.bias is not None:
            pointer_params.append(self.W_ref2.bias)
        pointer_params.append(self.v_1)

        # 모든 어텐션 관련 파라미터
        self.all_attention_params = list(
            self.ex_embedding.parameters()) + attention_params_1 + attention_params_2 + attention_params_3 + pointer_params

        self._initialize_weights(params["init_min"], params["init_max"])
        self.use_logit_clipping = params["use_logit_clipping"]
        self.C = params["C"]
        self.T = params["T"]
        self.n_glimpse = params["n_glimpse"]
        self.job_selecter = Categorical()
        self.lb_records = [[],[],[],[],[],[]]
        self.makespan_records = []





    def get_jssp_instance(self, instance): # 훈련해야할 instance를 에이전트가 참조(등록)하는 코드
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
                pass
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


    def get_critical_check(self, scheduler, mask):
        available_operations = mask
        avail_nodes = np.array(available_operations)
        avail_nodes_indices = np.where(avail_nodes == 1)[0].tolist() # 현재 시점에 가능한 operation들의 모임이다.
        scheduler.check_avail_ops(avail_nodes_indices)


####
    def forward(self, x, device, scheduler_list, num_job, num_machine, old_sequence = None, train = True, q_update = False):

        node_features, heterogeneous_edges = x
        node_features = torch.tensor(node_features).to(device).float()
        pi_list, log_ps = [], []

        log_probabilities = list()


        sample_space = [[j for i in range(num_machine)] for j in range(num_job)]
        sample_space = torch.tensor(sample_space).view(-1)


        edge_loss, node_loss, loss_kld, mean_feature, features, z = self.Latent.calculate_loss(node_features, heterogeneous_edges, train)


        batch = features.shape[0]
        num_operations = features.shape[1]
        """
        이 위에 까지가 Encoder
        이 아래 부터는 Decoder
    
        """

        h_pi_t_minus_one = self.v_1.unsqueeze(0).repeat(batch, 1).unsqueeze(0).to(device) # 이녀석이 s.o.s(start of singal)에 해당
        mask1_debug, mask2_debug = self.init_mask()

        batch_size = h_pi_t_minus_one.shape[1]
        #print(h_pi_t_minus_one.shape)

        if old_sequence != None:
            old_sequence = torch.tensor(old_sequence).long().to(device)
        next_operation_indices = list()

        action_sequences = torch.zeros([batch_size, num_operations, 4]).to(device)
        for i in range(num_operations):
            est_placeholder = mask2_debug.clone().to(device)
            fin_placeholder = mask2_debug.clone().to(device)

            mask1_debug = mask1_debug.reshape(batch_size, -1)
            mask2_debug = mask2_debug.reshape(batch_size, -1)
            empty_zero = torch.zeros(batch_size, num_operations).to(device)
            empty_zero2 = torch.zeros(batch_size, num_operations).to(device)


            if i == 0:
                """
                Earliest Start Time (est_placeholder)
                Earliest Finish Time (fin_placeholder) 확인하는 로직
                i == 0일 때는 아직 선택된 operation이 없으므로,
                adaptive_run에 선택된 변수(i)에 대한 정보가 없음
                
                """
                cp_list = []


                for nb in range(batch_size):
                    c_max, est, fin, critical_path, critical_path2 = scheduler_list[nb].adaptive_run(est_placeholder[nb], fin_placeholder[nb])
                    #print(empty_zero.shape, critical_path.shape)
                    empty_zero[nb, :] = torch.tensor(critical_path.reshape(-1)).to(device)# 안중요
                    empty_zero2[nb, :] = torch.tensor(critical_path2.reshape(-1)).to(device)  # 안중요
                    est_placeholder[nb] = est
                    fin_placeholder[nb] = fin



            else:
                """
                Earliest Start Time (est_placeholder)
                Earliest Finish Time (fin_placeholder) 확인하는 로직
                i == 0일 때는 아직 선택된 operation이 없으므로,
                adaptive_run에 선택된 변수(i)에 대한 정보는 이전에 선택된 index(next_operation_index)에서 추출

                """
                cp_list = []
                for nb in range(batch_size):
                    next_b = next_job[nb].item()
                    c_max, est, fin, critical_path, critical_path2 = scheduler_list[nb].adaptive_run(est_placeholder[nb], fin_placeholder[nb], i = next_b) # next_b는 이전 스텝에서 선택된 Job이고, Adaptive Run이라는 것은 선택된 Job에 따라 update한 다음에 EST, EFIN을 구하라는 의미
                    empty_zero[nb, :]  = torch.tensor(critical_path.reshape(-1)).to(device)
                    empty_zero2[nb, :] = torch.tensor(critical_path2.reshape(-1)).to(device)  # 안중요
                    est_placeholder[nb] = est
                    fin_placeholder[nb] = fin
                    """
                    
                    Branch and Cut 로직에 따라 masking을 수행함
                    모두 다 masking 처리할 수도 있으므로, 모두다 masking할 경우에는 mask로 복원 (if 1 not in mask)
                    
                    """

            est_placeholder = est_placeholder.reshape(batch_size, -1).unsqueeze(2)
            fin_placeholder = fin_placeholder.reshape(batch_size, -1).unsqueeze(2)
            empty_zero = empty_zero.unsqueeze(2)
            empty_zero2 = empty_zero2.unsqueeze(2)

            r_temp_raw = torch.concat([est_placeholder, fin_placeholder, empty_zero, empty_zero2], dim=2)  # extended node embedding을 만드는 부분(z_t_i에 해당)
            r_temp = r_temp_raw.reshape([batch*num_operations, -1])
            r_temp = self.ex_embedding(r_temp)
            r_temp = r_temp.reshape([batch, num_operations, -1])


            ref = torch.concat([features, r_temp], dim=2)
            if self.params['w_representation_learning'] == True:
                h_c = self.decoder(z.reshape(1, batch_size, -1).detach(), h_pi_t_minus_one.reshape(1, batch_size, -1)) # decoding 만드는 부분
            else:
                h_c = self.decoder(z.reshape(1, batch_size, -1),
                                   h_pi_t_minus_one.reshape(1, batch_size, -1))  # decoding 만드는 부분
            query = h_c.squeeze(0)
            """
            Query를 만들때에는 이전 단계의 query와 extended node embedding을 가지고 만든다

            """
            #print(query.shape, ref.shape)
            query = self.glimpse(query, ref, mask2_debug)  # 보는 부분 /  multi-head attention 부분 (mask2는 보는 masking)
            logits = self.pointer(query, ref, mask1_debug) # 선택하는 부분 / logit 구하는 부분 (#mask1은 선택하는 masking)

            y_hard, y_soft = gumbel_softmax_hard(logits, tau=1, dim=-1)
            mask = (y_hard == 1)
            selected_action = torch.einsum('bnk, bn -> bk', r_temp_raw, y_hard)

            # selected_action = r_temp_raw[mask]*y_hard[mask].reshape(batch_size,1)
            # y_soft에서 마스크가 True인 요소만 추출


            action_sequences[:, i, :] = selected_action

            hard_log_p = torch.log(y_hard)
            if old_sequence == None:
                next_operation_index = self.job_selecter(hard_log_p)
            else:
                next_operation_index = old_sequence[i, :]

            selected_elements = y_soft[mask]
            log_p = torch.log(selected_elements+1e-10)
            log_probabilities.append(log_p)
            sample_space = sample_space.to(device)
            next_job = sample_space[next_operation_index].to(device)
            mask1_debug, mask2_debug = self.update_mask(next_job.tolist()) # update masking을 수행해주는

            batch_indices = torch.arange(features.size(0))
            h_pi_t_minus_one = features[batch_indices, next_operation_index]


            #h_pi_t_minus_one = torch.gather(input=features, dim=1, index=next_operation_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mean_feature.shape[2])).squeeze(1).unsqueeze(0)  # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
            next_operation_indices.append(next_operation_index.tolist())
            pi_list.append(next_job)

        if train == True:
            if self.params['w_representation_learning']==True:
                if q_update == True:
                    q = self.critic(z.detach(), action_sequences.detach())
                else:
                    q = self.critic(z.detach(), action_sequences)


            else:
                q = self.critic(z, action_sequences)

        else:
            q = None
            q_for_critic_update = None



        pi = torch.stack(pi_list, dim=1)
        log_probabilities = torch.stack(log_probabilities, dim=1)

        ll = log_probabilities.sum(dim=1)    # 각 solution element의 log probability를 더하는 방식
        return pi, ll, next_operation_indices, edge_loss, node_loss, loss_kld, q

    def glimpse(self, query, ref, mask0):
        """
        query는 decoder의 출력
        ref는   encoder의 출력
        """
        batch_size = query.shape[0]

        dk = self.params["n_hidden"]/self.n_multi_head
        for m in range(self.n_multi_head):
            u1 = self.W_q[m](query).unsqueeze(1)
            u2 = self.W_ref[m](ref.reshape(ref.shape[0]*ref.shape[1],-1))                             # u2: (batch, 128, block_num)
            u2 = u2.reshape(ref.shape[0], ref.shape[1], -1)
            u2 = u2.permute(0, 2, 1)
            u = torch.bmm(u1, u2)/dk**0.5
            v = ref@self.Vec[m]
            u = u.squeeze(1).masked_fill(mask0 == 0, -1e8)
            a = F.softmax(u, dim=1)
            if m == 0:
                g = torch.bmm(a.unsqueeze(1), v).squeeze(1)
            else:
                g += torch.bmm(a.unsqueeze(1), v).squeeze(1)

        dk = self.params["n_hidden"] / self.n_multi_head
        for m in range(self.n_multi_head):
            u1 = self.W_q[m](query).unsqueeze(1)
            u2 = self.W_ref[m](ref.reshape(ref.shape[0] * ref.shape[1], -1))  # u2: (batch, 128, block_num)
            u2 = u2.reshape(ref.shape[0], ref.shape[1], -1)
            u2 = u2.permute(0, 2, 1)
            u = torch.bmm(u1, u2) / dk ** 0.5
            v = ref @ self.Vec[m]
            u = u.squeeze(1).masked_fill(mask0 == 0, -1e8)
            a = F.softmax(u, dim=1)
            if m == 0:
                g = torch.bmm(a.unsqueeze(1), v).squeeze(1)
            else:
                g += torch.bmm(a.unsqueeze(1), v).squeeze(1)


        dk = self.params["n_hidden"]/self.n_multi_head
        for m in range(self.n_multi_head):
            u1 = self.W_q[m](query).unsqueeze(1)
            u2 = self.W_ref[m](ref.reshape(ref.shape[0]*ref.shape[1],-1))                             # u2: (batch, 128, block_num)
            u2 = u2.reshape(ref.shape[0], ref.shape[1], -1)
            u2 = u2.permute(0, 2, 1)
            u = torch.bmm(u1, u2)/dk**0.5
            v = ref@self.Vec[m]
            u = u.squeeze(1).masked_fill(mask0 == 0, -1e8)
            a = F.softmax(u, dim=1)
            if m == 0:
                g = torch.bmm(a.unsqueeze(1), v).squeeze(1)
            else:
                g += torch.bmm(a.unsqueeze(1), v).squeeze(1)

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
