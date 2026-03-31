import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cfg
from copy import deepcopy
cfg = cfg.get_cfg()
from model import GCRN
from latent import LatentModel
device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')



class Categorical(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, log_p):
        return torch.multinomial(log_p.exp(), 1).long().squeeze(1)

class Greedy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, log_p):
        #print(log_p.shape)
        return torch.argmax(log_p, -1)#.long().squeeze(1)


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
    def forward(self, x, visualize = False):
        x = F.elu(self.fcn1(x))
        x = F.elu(self.fcn2(x))

        if visualize == False:
            x = F.elu(self.fcn3(x))
            x = F.elu(self.fcn4(x))
            x = self.fcn5(x)
            return x
        else:
            x = F.elu(self.fcn3(x))
            h = F.elu(self.fcn4(x))
            x = self.fcn5(h)
            return x, h



class PtrNet1(nn.Module):
    def __init__(self, params):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_multi_head = params["n_multi_head"]
        self.params = params
        self.k_hop = params["k_hop"]

        num_edge_cat = 3
        z_dim = params["n_hidden"]
        self.critic = Critic(z_dim)
        self.Latent = LatentModel(z_dim=z_dim, params = params).to(device)
        augmented_hidden_size = params["n_hidden"]

        if cfg.state_feature_selection==False:
            self.ex_embedding = ExEmbedding(raw_feature_size=6, feature_size= params["n_hidden"])
        else:
            self.ex_embedding = ExEmbedding(raw_feature_size=4, feature_size=params["n_hidden"])

        # Vec 파라미터 리스트 생성 (문제 없음)
        self.W_v = nn.ParameterList([nn.Parameter(torch.FloatTensor(2 * params["n_hidden"], 2 * params["n_hidden"]))for _ in range(self.n_multi_head)])
        self.W_q = nn.ModuleList([nn.Linear(2 * params["n_hidden"],  params["n_hidden"]+params["ex_embedding_size"], bias=False).to(device)for _ in range(self.n_multi_head)])
        self.W_k = nn.ModuleList([nn.Linear(2 * params["n_hidden"],  params["n_hidden"]+params["ex_embedding_size"],bias=False).to(device) for _ in range(self.n_multi_head)])
        self.W_o = nn.ModuleList([nn.Linear(2 * params["n_hidden"], 2 * params["n_hidden"], bias=False).to(device) for _ in range(self.n_multi_head)])

        # 마지막 포인터 네트워크 관련 파라미터는 그대로 유지
        self.Vec2 = nn.Parameter(torch.FloatTensor(2 *  params["n_hidden"]))
        self.W_q2 = nn.Linear(2 *  params["n_hidden"], params["n_hidden"]+params["ex_embedding_size2"], bias=False)
        self.W_ref2 = nn.Linear(2 *  params["n_hidden"],  params["n_hidden"]+params["ex_embedding_size2"],  bias=False)
        self.v_1 = nn.Parameter(torch.FloatTensor(params["n_hidden"]))

        # 파라미터 목록 생성 방식도 변경
        # 모든 어텐션 관련 파라미터를 각 모듈에서 parameters() 메소드로 추출
        attention_params_1 = list(self.W_v) + [p for m in self.W_q for p in m.parameters()] + [p for m in self.W_k for
                                                                                               p in m.parameters()] +[p for m in self.W_o for
                                                                                               p in m.parameters()]
        # attention_params_2 = list(self.Vec3) + [p for m in self.W_q3 for p in m.parameters()] + [p for m in self.W_ref3
        #                                                                                          for p in
        #                                                                                          m.parameters()]
        # attention_params_3 = list(self.Vec4) + [p for m in self.W_q4 for p in m.parameters()] + [p for m in self.W_ref4
        #                                                                                          for p in
        #                                                                                          m.parameters()]

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
            self.ex_embedding.parameters()) + attention_params_1 + pointer_params

        self._initialize_weights(params["init_min"], params["init_max"])
        self.use_logit_clipping = params["use_logit_clipping"]
        self.C = params["C"]
        self.T = params["T"]
        self.n_glimpse = params["n_glimpse"]
        self.job_selecter = Categorical()
        self.job_selecter_greedy = Greedy()
        self.lb_records = [[],[],[],[],[],[]]
        self.makespan_records = []
        self.log_alpha = nn.Parameter(torch.tensor(0.0001))





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
    def forward_latent(self, x, device, scheduler_list, num_job, num_machine, old_sequence=None, train=True,
                old_sequence_in_ops=None, visualize = False):
        node_features, heterogeneous_edges = x
        node_features = torch.tensor(node_features).to(device).float()
        pi_list, log_ps = [], []
        log_probabilities = list()
        #sample_space = [[j for i in range(num_machine)
        edge_loss, node_loss, loss_kld, mean_feature, features, z = self.Latent.calculate_loss(node_features,
                                                                                               heterogeneous_edges,
                                                                                               train)
        if visualize == False:
            return edge_loss,node_loss,loss_kld
        else:
            baselines, h = self.critic(z, visualize = True)
            return z, baselines, h

    def forward_visualize(self, x, device, scheduler_list, num_job, num_machine, old_sequence=None, train=True, old_sequence_in_ops=None):
        node_features, heterogeneous_edges = x
        node_features = torch.tensor(node_features).to(device).float()
        pi_list, log_ps = [], []
        log_probabilities = list()
        sample_space = [[j for i in range(num_machine)] for j in range(num_job)]
        sample_space = torch.tensor(sample_space).view(-1)
        mean_feature, features, z, z_mean_post = self.Latent.calculate_feature_embedding(node_features,
                                                                                               heterogeneous_edges,
                                                                                               train=False)

        baselines, h = self.critic(z.detach(), visualize = True)
        batch = features.shape[0]
        num_operations = features.shape[1]
        """
        이 위에 까지가 Encoder
        이 아래 부터는 Decoder
        """

        h_pi_t_minus_one = self.v_1.unsqueeze(0).repeat(batch, 1).unsqueeze(0).to(device)  # 이녀석이 s.o.s(start of singal)에 해당
        mask1_debug, mask2_debug = self.init_mask()

        batch_size = h_pi_t_minus_one.shape[1]

        if old_sequence != None:
            old_sequence = torch.tensor(old_sequence).long().to(device)
        next_operation_indices = list()
        lb_records = [[], [], [], [], [], []]

        for i in range(num_operations):
            est_placeholder = mask2_debug.clone().to(device)
            fin_placeholder = mask2_debug.clone().to(device)
            mwkr_placeholder1 = mask2_debug.clone().to(device)
            mwkr_placeholder2 = mask2_debug.clone().to(device)

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
                    c_max, est, fin, critical_path, critical_path2, mwkr1, mwkr2 = scheduler_list[nb].adaptive_run(
                        est_placeholder[nb],
                        fin_placeholder[nb],
                        mwkr_placeholder1[nb],
                        mwkr_placeholder2[nb],
                        )
                    # print(empty_zero.shape, critical_path.shape)
                    empty_zero[nb, :] = torch.tensor(critical_path.reshape(-1)).to(device)  # 안중요
                    empty_zero2[nb, :] = torch.tensor(critical_path2.reshape(-1)).to(device)  # 안중요
                    est_placeholder[nb] = est
                    fin_placeholder[nb] = fin
                    mwkr_placeholder1[nb] = mwkr1
                    mwkr_placeholder2[nb] = mwkr2



            else:
                """
                Earliest Start Time (est_placeholder)
                Earliest Finish Time (fin_placeholder) 확인하는 로직
                i == 0일 때는 아직 선택된 operation이 없으므로,
                adaptive_run에 선택된 변수(i)에 대한 정보는 이전에 선택된 index(next_operation_index)에서 추출

                """
                cp_list = []
                for nb in range(batch_size):
                    if old_sequence != None:
                        # print(old_sequence.shape)
                        next_b = old_sequence[nb, i].item()
                    else:
                        next_b = next_job[nb].item()
                    c_max, est, fin, critical_path, critical_path2, mwkr1, mwkr2 = scheduler_list[nb].adaptive_run(
                        est_placeholder[nb], fin_placeholder[nb],
                        mwkr_placeholder1[nb],
                        mwkr_placeholder2[nb],
                        i=next_b)  # next_b는 이전 스텝에서 선택된 Job이고, Adaptive Run이라는 것은 선택된 Job에 따라 update한 다음에 EST, EFIN을 구하라는 의미

                    empty_zero[nb, :] = torch.tensor(critical_path.reshape(-1)).to(device)
                    empty_zero2[nb, :] = torch.tensor(critical_path2.reshape(-1)).to(device)  # 안중요
                    est_placeholder[nb] = est
                    fin_placeholder[nb] = fin
                    # print("전",est[0])
                    # print("후",est_placeholder[nb][0])
                    # print('====================')
                    mwkr_placeholder1[nb] = mwkr1
                    mwkr_placeholder2[nb] = mwkr2
                    """

                    Branch and Cut 로직에 따라 masking을 수행함
                    모두 다 masking 처리할 수도 있으므로, 모두다 masking할 경우에는 mask로 복원 (if 1 not in mask)

                    """

            est_placeholder = est_placeholder.reshape(batch_size, -1).unsqueeze(2)
            fin_placeholder = fin_placeholder.reshape(batch_size, -1).unsqueeze(2)
            mwkr_placeholder1 = mwkr_placeholder1.reshape(batch_size, -1).unsqueeze(2)
            mwkr_placeholder2 = mwkr_placeholder2.reshape(batch_size, -1).unsqueeze(2)
            empty_zero = empty_zero.unsqueeze(2)
            empty_zero2 = empty_zero2.unsqueeze(2)
            # print(est_placeholder.shape, mwkr_placeholder2.shape)

            r_temp = torch.concat([est_placeholder, fin_placeholder, empty_zero, empty_zero2, mwkr_placeholder1, mwkr_placeholder2], dim=2)  # extended node embedding을 만드는 부분(z_t_i에 해당)

            r_temp = r_temp.reshape([batch * num_operations, -1])
            r_temp = self.ex_embedding(r_temp)
            r_temp = r_temp.reshape([batch, num_operations, -1])
            ref = torch.concat([features, r_temp], dim=2)

            if self.params['w_representation_learning'] == True:
                h_c = self.decoder(z.reshape(1, batch_size, -1).detach(),
                                   h_pi_t_minus_one.reshape(1, batch_size, -1))  # decoding 만드는 부분
            else:
                h_c = self.decoder(z.reshape(1, batch_size, -1),
                                   h_pi_t_minus_one.reshape(1, batch_size, -1))  # decoding 만드는 부분
            query = h_c.squeeze(0)
            """
            Query를 만들때에는 이전 단계의 query와 extended node embedding을 가지고 만든다

            """
            query = self.glimpse(query, ref, mask2_debug)  # 보는 부분 /  multi-head attention 부분 (mask2는 보는 masking)
            logits = self.pointer(query, ref, mask1_debug)  # 선택하는 부분 / logit 구하는 부분 (#mask1은 선택하는 masking)

            cp_list = torch.tensor(cp_list)
            # print(cp_list.shape)

            log_p = torch.log_softmax(logits / self.T, dim=-1)  # log_softmax로 구하는 부분

            if old_sequence == None:
                if train == True:
                    next_operation_index = self.job_selecter(log_p)
                else:
                    next_operation_index = self.job_selecter_greedy(log_p)
            else:
                next_operation_index = torch.tensor(old_sequence_in_ops).to(device).long()[i, :]

            log_probabilities.append(log_p.gather(1, next_operation_index.unsqueeze(1)))
            sample_space = sample_space.to(device)
            next_job = sample_space[next_operation_index].to(device)
            mask1_debug, mask2_debug = self.update_mask(next_job.tolist())  # update masking을 수행해주는

            batch_indices = torch.arange(features.size(0))
            h_pi_t_minus_one = features[batch_indices, next_operation_index]

            # h_pi_t_minus_one = torch.gather(input=features, dim=1, index=next_operation_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mean_feature.shape[2])).squeeze(1).unsqueeze(0)  # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
            next_operation_indices.append(next_operation_index.tolist())
            pi_list.append(next_job)

        pi = torch.stack(pi_list, dim=1)
        return z, baselines, h, pi, mean_feature, z_mean_post

    def forward(self, x, device, scheduler_list, num_job, num_machine, old_sequence = None, train = True, old_sequence_in_ops=None):
        node_features, heterogeneous_edges = x
        node_features = torch.tensor(node_features).to(device).float()
        pi_list, log_ps = [], []
        log_probabilities = list()
        sample_space = [[j for i in range(num_machine)] for j in range(num_job)]
        sample_space = torch.tensor(sample_space).view(-1)
        edge_loss, node_loss, loss_kld, mean_feature, features, z = self.Latent.calculate_loss(node_features, heterogeneous_edges, train)

        if self.params['w_representation_learning'] == True:
            baselines = self.critic(z.detach())
        else:
            baselines = self.critic(z)

        batch = features.shape[0]
        num_operations = features.shape[1]
        """
        이 위에 까지가 Encoder
        이 아래 부터는 Decoder
    
        """

        h_pi_t_minus_one = self.v_1.unsqueeze(0).repeat(batch, 1).unsqueeze(0).to(device) # 이녀석이 s.o.s(start of singal)에 해당
        mask1_debug, mask2_debug = self.init_mask()

        batch_size = h_pi_t_minus_one.shape[1]

        if old_sequence != None:
            old_sequence = torch.tensor(old_sequence).long().to(device)
        next_operation_indices = list()
        lb_records = [[],[],[],[],[],[]]

        for i in range(num_operations):
            est_placeholder = mask2_debug.clone().to(device)
            fin_placeholder = mask2_debug.clone().to(device)
            mwkr_placeholder1 = mask2_debug.clone().to(device)
            mwkr_placeholder2 = mask2_debug.clone().to(device)

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
                    c_max, est, fin, critical_path, critical_path2,mwkr1,mwkr2 = scheduler_list[nb].adaptive_run(est_placeholder[nb],
                                                                                                     fin_placeholder[nb],
                                                                                                     mwkr_placeholder1[nb],
                                                                                                     mwkr_placeholder2[nb],
                                                                                                     )
                    #print(empty_zero.shape, critical_path.shape)
                    empty_zero[nb, :] = torch.tensor(critical_path.reshape(-1)).to(device)# 안중요
                    empty_zero2[nb, :] = torch.tensor(critical_path2.reshape(-1)).to(device)  # 안중요
                    est_placeholder[nb] = est
                    fin_placeholder[nb] = fin
                    mwkr_placeholder1[nb] = mwkr1
                    mwkr_placeholder2[nb] = mwkr2



            else:
                """
                Earliest Start Time (est_placeholder)
                Earliest Finish Time (fin_placeholder) 확인하는 로직
                i == 0일 때는 아직 선택된 operation이 없으므로,
                adaptive_run에 선택된 변수(i)에 대한 정보는 이전에 선택된 index(next_operation_index)에서 추출

                """
                cp_list = []
                for nb in range(batch_size):
                    if old_sequence != None:
                        #print(old_sequence.shape)
                        next_b = old_sequence[nb, i].item()
                    else:
                        next_b = next_job[nb].item()
                    c_max, est, fin, critical_path, critical_path2,mwkr1,mwkr2  = scheduler_list[nb].adaptive_run(
                        est_placeholder[nb], fin_placeholder[nb],
                        mwkr_placeholder1[nb],
                        mwkr_placeholder2[nb],
                        i = next_b) # next_b는 이전 스텝에서 선택된 Job이고, Adaptive Run이라는 것은 선택된 Job에 따라 update한 다음에 EST, EFIN을 구하라는 의미

                    empty_zero[nb, :]  = torch.tensor(critical_path.reshape(-1)).to(device)
                    empty_zero2[nb, :] = torch.tensor(critical_path2.reshape(-1)).to(device)  # 안중요
                    est_placeholder[nb] = est
                    fin_placeholder[nb] = fin
                    # print("전",est[0])
                    # print("후",est_placeholder[nb][0])
                    # print('====================')
                    mwkr_placeholder1[nb] = mwkr1
                    mwkr_placeholder2[nb] = mwkr2
                    """
                    
                    Branch and Cut 로직에 따라 masking을 수행함
                    모두 다 masking 처리할 수도 있으므로, 모두다 masking할 경우에는 mask로 복원 (if 1 not in mask)
                    
                    """

            est_placeholder = est_placeholder.reshape(batch_size, -1).unsqueeze(2)
            fin_placeholder = fin_placeholder.reshape(batch_size, -1).unsqueeze(2)
            mwkr_placeholder1 = mwkr_placeholder1.reshape(batch_size, -1).unsqueeze(2)
            mwkr_placeholder2 = mwkr_placeholder2.reshape(batch_size, -1).unsqueeze(2)
            empty_zero = empty_zero.unsqueeze(2)
            empty_zero2 = empty_zero2.unsqueeze(2)
           # print(est_placeholder.shape, mwkr_placeholder2.shape)

            if cfg.state_feature_selection==False:
                r_temp = torch.concat([est_placeholder, fin_placeholder, empty_zero, empty_zero2, mwkr_placeholder1,mwkr_placeholder2], dim=2)  # extended node embedding을 만드는 부분(z_t_i에 해당)
            else:
                if cfg.state_feature_group=='group1':
                    r_temp = torch.concat([est_placeholder, fin_placeholder, empty_zero, empty_zero2, mwkr_placeholder1, mwkr_placeholder2], dim=2)
                elif cfg.state_feature_group == 'group2':
                    r_temp = torch.concat([est_placeholder, fin_placeholder, mwkr_placeholder1,  mwkr_placeholder2], dim=2)
                elif cfg.state_feature_group == 'group3':
                    r_temp = torch.concat([est_placeholder, fin_placeholder, empty_zero, empty_zero2], dim=2)

            r_temp = r_temp.reshape([batch*num_operations, -1])
            r_temp = self.ex_embedding(r_temp)
            r_temp = r_temp.reshape([batch, num_operations, -1])
            ref = torch.concat([features, r_temp], dim=2)

            if self.params['w_representation_learning'] == True:
                h_c = self.decoder(z.reshape(1, batch_size, -1).detach(), h_pi_t_minus_one.reshape(1, batch_size, -1))  # decoding 만드는 부분
            else:
                h_c = self.decoder(z.reshape(1, batch_size, -1), h_pi_t_minus_one.reshape(1, batch_size, -1))  # decoding 만드는 부분
            query = h_c.squeeze(0)
            """
            Query를 만들때에는 이전 단계의 query와 extended node embedding을 가지고 만든다

            """
            query = self.glimpse(query, ref, mask2_debug)  # 보는 부분 /  multi-head attention 부분 (mask2는 보는 masking)
            logits = self.pointer(query, ref, mask1_debug) # 선택하는 부분 / logit 구하는 부분 (#mask1은 선택하는 masking)

            cp_list = torch.tensor(cp_list)
            #print(cp_list.shape)

            log_p = torch.log_softmax(logits / self.T, dim=-1) # log_softmax로 구하는 부분

            if old_sequence == None:
                if train == True:
                    next_operation_index = self.job_selecter(log_p)
                else:
                    next_operation_index = self.job_selecter_greedy(log_p)
            else:
                next_operation_index = torch.tensor(old_sequence_in_ops).to(device).long()[i, :]



            log_probabilities.append(log_p.gather(1, next_operation_index.unsqueeze(1)))
            sample_space = sample_space.to(device)
            next_job = sample_space[next_operation_index].to(device)
            mask1_debug, mask2_debug = self.update_mask(next_job.tolist()) # update masking을 수행해주는

            batch_indices = torch.arange(features.size(0))
            h_pi_t_minus_one = features[batch_indices, next_operation_index]



            #h_pi_t_minus_one = torch.gather(input=features, dim=1, index=next_operation_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, mean_feature.shape[2])).squeeze(1).unsqueeze(0)  # 다음 sequence의 input은 encoder의 output 중에서 현재 sequence에 해당하는 embedding이 된다.
            next_operation_indices.append(next_operation_index.tolist())
            pi_list.append(next_job)



        pi = torch.stack(pi_list, dim=1)
        log_probabilities = torch.stack(log_probabilities, dim=1)
        ll = log_probabilities.sum(dim=1)    # 각 solution element의 log probability를 더하는 방식

        return pi, ll, next_operation_indices, edge_loss, node_loss, loss_kld, baselines

    def glimpse(self, query, ref, mask0):
        """
        query는 decoder의 출력
        ref는   encoder의 출력
        """
        dk = self.params["n_hidden"]/self.n_multi_head
        for dd in range(2):
            for m in range(self.n_multi_head):
                u1 = self.W_q[m](query).unsqueeze(1)
                u2 = self.W_k[m](ref.reshape(ref.shape[0]*ref.shape[1],-1))                             # u2: (batch, 128, block_num)
                u2 = u2.reshape(ref.shape[0], ref.shape[1], -1)
                u2 = u2.permute(0, 2, 1)
                u = torch.bmm(u1, u2)/dk**0.5
                v = ref@self.W_v[m]
                u = u.squeeze(1).masked_fill(mask0 == 0, -1e8)
                a = F.softmax(u, dim=1)
                if m == 0:
                    g = torch.bmm(a.unsqueeze(1), v).squeeze(1)#/self.n_multi_head
                    #print(g.shape, self.W_o.shape)
                else:
                    g += torch.bmm(a.unsqueeze(1), v).squeeze(1)#/self.n_multi_head
            query = g

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
        #print(h_bar.shape, h_t_minus_one.shape)
        return torch.concat([h_bar, h_t_minus_one], dim =2)
