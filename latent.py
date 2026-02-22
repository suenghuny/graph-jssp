import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import ContinuousBernoulli
import numpy as np
from utils import *
import cfg
from copy import deepcopy
cfg = cfg.get_cfg()
from model import GCRN
device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

class FixedGaussian(nn.Module):
    """
    Fixed diagonal gaussian distribution.
    """

    def __init__(self, output_dim, std):
        super().__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):

        mean = torch.zeros(x.size(0), self.output_dim, device=x.device)
        std = torch.ones(x.size(0), self.output_dim, device=x.device).mul_(self.std)
        return mean, std


class Gaussian(nn.Module):
    """
    Diagonal gaussian distribution with state dependent variances.
    """

    def __init__(self, input_dim, output_dim, hidden_units=(256, 256)):
        super().__init__()


        self.net = build_mlp(
            input_dim=input_dim,
            output_dim=output_dim*2,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(0.2),
        ).apply(initialize_weight)

        self.net_mean = build_mlp(
            input_dim=output_dim,
            output_dim=output_dim,
            hidden_units=[128, 128],
            hidden_activation=nn.LeakyReLU(0.2),
        ).apply(initialize_weight)

        self.net_std = build_mlp(
            input_dim=output_dim,
            output_dim=output_dim,
            hidden_units=[128, 128],
            hidden_activation=nn.LeakyReLU(0.2),
        ).apply(initialize_weight)


    def forward(self, x):
        x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        mean, std = self.net_mean(mean), self.net_std(std)
        std = F.softplus(std) + 1e-5
        return mean, std


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n_multi_head = params["n_multi_head"]
        if cfg.feature_selection_mode == True:
            self.Embedding = nn.Linear(5, params["n_hidden"], bias=False).to(device)  # 그림 상에서 Encoder에 FF(feedforward)라고 써져있는 부분
        else:
            self.Embedding = nn.Linear(6, params["n_hidden"], bias=False).to(
                device)  # 그림 상에서 Encoder에 FF(feedforward)라고 써져있는 부분
        self.params = params
        self.k_hop = params["k_hop"]
        self.aggr = params["aggr"]
        num_edge_cat = 3

        self.GraphEmbedding = GCRN(
                                   feature_size =  params["n_hidden"],
                                   graph_embedding_size= params["graph_embedding_size"],
                                   embedding_size =  params["n_hidden"],
                                   layers =  params["layers"],
                                   alpha =  params["alpha"],
                                   n_multi_head = params["n_multi_head"],
                                   num_edge_cat = num_edge_cat,
                                   aggr=params["aggr"]
                                   ).to(device)
        self.GraphEmbedding1 = GCRN(feature_size =  params["n_hidden"],
                                   graph_embedding_size= params["graph_embedding_size"],
                                   embedding_size =  params["n_hidden"],
                                    n_multi_head=params["n_multi_head"],
                                    layers =  params["layers"],
                                    alpha=params["alpha"],
                                    num_edge_cat = num_edge_cat,
                                    aggr=params["aggr"]
                                    ).to(device)


    def forward(self, node_features, heterogeneous_edges):
        batch = node_features.shape[0]
        operation_num = node_features.shape[1]-2
        node_num = node_features.shape[1]
        node_reshaped_features = node_features.reshape(batch * node_num, -1)
        node_embedding = self.Embedding(node_reshaped_features)

        node_embedding = node_embedding.reshape(batch, node_num, -1)
        if self.k_hop == 1:
            enc_h, edge_cats = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
        if self.k_hop == 2:
            enc_h, edge_cats = self.GraphEmbedding(heterogeneous_edges, node_embedding,  mini_batch = True)
            enc_h, edge_cats = self.GraphEmbedding1(heterogeneous_edges, enc_h, mini_batch=True, final = True)

        if self.aggr == 'mean':
            h_mean = enc_h.mean(dim=1)
            h = h_mean
        elif self.aggr == 'mean_max':
            h_mean = enc_h.mean(dim=1)
            h_max = enc_h.max(dim=1)[0]
            h = torch.concat([h_mean, h_max], dim =1)
        elif self.aggr == 'mean_max_std':
            h_mean = enc_h.mean(dim=1)
            h_max = enc_h.max(dim=1)[0]
            h_std = enc_h.std(dim=1)
            h = torch.concat([h_mean, h_max, h_std], dim=1)

            #


        enc_h = enc_h[:, :-2]                 # dummy node(source, sink)는 제외한다.
        return h, enc_h, edge_cats




#
# class GraphVAEDecoder(nn.Module):
#     """
#     GraphVAE decoder that takes a latent vector z and outputs a probabilistic fully-connected graph.
#     The graph consists of:
#     - Adjacency matrix A ∈ [0,1]^(k×k)
#     - Edge attribute tensor E ∈ [0,1]^(k×k×de)
#     - Node attribute matrix F ∈ [0,1]^(k×dn)
#
#     Where:
#     - k is the maximum number of nodes
#     - de is the number of edge types
#     - dn is the number of node types
#     """
#
#     def __init__(self, latent_dim, max_nodes=100, edge_types=3, node_types=4, hidden_dims=[128, 256, 512]):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.max_nodes = max_nodes
#         self.edge_types = edge_types
#         self.node_types = node_types
#
#
#
#         # Fully connected layers
#         layers = []
#         prev_dim = latent_dim
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(nn.BatchNorm1d(hidden_dim))
#             layers.append(nn.ReLU())
#             prev_dim = hidden_dim
#
#         self.fc_layers = nn.Sequential(*layers)
#
#         # Output layers for adjacency matrix, edge attributes, and node attributes
#         self.fc_edge = nn.Linear(hidden_dims[-1], max_nodes * max_nodes * edge_types)
#         self.fc_node = nn.Linear(hidden_dims[-1], max_nodes * node_types)
#
#     def forward(self, z):
#         """
#         Forward pass through the decoder.
#
#         Args:
#             z: Latent vectors of shape [batch_size, latent_dim]
#
#         Returns:
#             adj_prob: Adjacency matrix probabilities [batch_size, max_nodes, max_nodes]
#             edge_prob: Edge type probabilities [batch_size, max_nodes, max_nodes, edge_types]
#             node_prob: Node type probabilities [batch_size, max_nodes, node_types]
#         """
#         batch_size = z.size(0)
#
#         # Compute feature representation
#         h = self.fc_layers(z)
#
#
#         # Compute edge type probabilities
#         edge_logits = self.fc_edge(h).view(batch_size, self.max_nodes, self.max_nodes, self.edge_types)
#         edge_prob = F.sigmoid(edge_logits)
#
#         # Compute node type probabilities
#         node_logits = self.fc_node(h).view(batch_size, self.max_nodes, self.node_types)
#         node_prob = F.sigmoid(node_logits)
#         return node_prob, edge_prob


class GraphVAEDecoder(nn.Module):
    """
    Alternative approach: Use 1D transposed convolutions to generate sequences,
    then reshape for graph structure.
    """

    def __init__(self, latent_dim, max_nodes=100, edge_types=3, node_types=4, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.edge_types = edge_types
        self.node_types = node_types

        # Initial expansion
        self.fc_initial = nn.Linear(latent_dim, 512)

        # 1D transposed convolutions for sequence generation
        self.conv1d_layers = nn.Sequential(
            # Start with a small sequence length and expand
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),  # Length: 2
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # Length: 4
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # Length: 8
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        # Calculate final sequence length after conv1d layers
        temp_length = 8  # From the conv1d operations above

        # Additional layers to reach desired length
        remaining_layers = []
        while temp_length < max_nodes * max_nodes:
            next_length = min(temp_length * 2, max_nodes * max_nodes)
            remaining_layers.extend([
                nn.ConvTranspose1d(64, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU()
            ])
            temp_length = next_length

        self.additional_conv1d = nn.Sequential(*remaining_layers)

        # Final output layers
        self.edge_output = nn.Conv1d(64, edge_types, kernel_size=1)
        self.node_output = nn.Sequential(
            nn.AdaptiveAvgPool1d(max_nodes),
            nn.Conv1d(64, node_types, kernel_size=1)
        )

    def forward(self, z):
        batch_size = z.size(0)
        #print('0', z.shape)
        # Initial transformation
        h = self.fc_initial(z).unsqueeze(-1)  # [batch_size, 512, 1]

        # Apply 1D transposed convolutions
        #print('1', h.shape)
        h = self.conv1d_layers(h)
        #print('2', h.shape)
        h = self.additional_conv1d(h)
        #print('3', h.shape)

        # Adjust sequence length for edges
        edge_seq_len = self.max_nodes * self.max_nodes
        if h.size(-1) != edge_seq_len:
            h_edges = F.interpolate(h, size=edge_seq_len, mode='linear', align_corners=False)
        else:
            h_edges = h

        #print('4', h_edges.shape)
        edge_logits = self.edge_output(h_edges)  # [batch_size, edge_types, max_nodes*max_nodes]
        #print('5', edge_logits.shape)
        edge_logits = edge_logits.view(batch_size, self.edge_types, self.max_nodes, self.max_nodes)
        edge_logits = edge_logits.permute(0, 2, 3, 1)  # [batch_size, max_nodes, max_nodes, edge_types]
        edge_prob = torch.sigmoid(edge_logits)

        # Generate node probabilities
        node_logits = self.node_output(h)  # [batch_size, node_types, max_nodes]
        node_logits = node_logits.permute(0, 2, 1)  # [batch_size, max_nodes, node_types]
        node_prob = torch.sigmoid(node_logits)

        return node_prob, edge_prob

class LatentModel(nn.Module):
    """
    Stochastic latent variable model to estimate latent dynamics and the reward.
    """

    def __init__(
        self,
        params,
        z_dim,
        hidden_units=(256, 256),
        ):
        super(LatentModel, self).__init__()

        self.current_num_edges = 100
        # p(z) = N(0, I)
        self.z_prior = FixedGaussian(z_dim, 1.0)

        # x = encoder(G)
        self.encoder = Encoder(params)
        # q(z | x)
        if params['aggr']== 'mean':
            aug_input_dim = z_dim
        elif params['aggr'] == 'mean_max':
            aug_input_dim = z_dim*2
        elif params['aggr'] == 'mean_max_std':
            aug_input_dim = z_dim * 3



        self.z_posterior = Gaussian(
            aug_input_dim,
            z_dim,
            hidden_units,
        )
        # p(G | z)
        if cfg.feature_selection_mode == True:
            self.decoder = GraphVAEDecoder(z_dim, edge_types=3, node_types=5, hidden_dims=[128, 256, 512])
        else:
            self.decoder = GraphVAEDecoder(z_dim, edge_types=3, node_types=6, hidden_dims=[128, 256, 512])


    def sample_prior(self, x):
        z_mean, z_std = self.z_prior(x)
        #z_std = torch.ones_like(z_mean)*0.1
        return z_mean, z_std

    def sample_posterior(self, features):
        # p(z1(0)) = N(0, I)
        z_mean, z_std = self.z_posterior(features)
        z = z_mean + torch.randn_like(z_std) * z_std
        return z_mean, z_std, z


    def calculate_loss(self, X, A, train = False):
        # Calculate the sequence of features.
        mean_feature, features, edge_cats = self.encoder(X, A)
        X = X.to(device)[:, :-2, :]
        edge_cats = edge_cats.to(device)[:, :-2, :-2, :]
        z_mean_pri, z_std_pri = self.sample_prior(mean_feature)


        z_mean_post, z_std_post, z = self.sample_posterior(mean_feature) # q(|G
        #print(z_mean_post.shape, z_std_post.shape, z_mean_pri.shape, z_std_pri.shape)
        loss_kld = calculate_kl_divergence(z_mean_post, z_std_post, z_mean_pri, z_std_pri).mean(dim=0).sum()
        if train == True:
            node_pred, edge_pred = self.decoder(z)
            node_pred = node_pred[:,:self.current_num_edges, :]
            edge_pred = edge_pred[:,:self.current_num_edges, :self.current_num_edges,:]
            edge_loss = edge_cats*torch.log(edge_pred)+(1-edge_cats)*torch.log(1-edge_pred)

            edge_loss = -edge_loss.mean()
            if cfg.loss_type == 'cross_entropy':
                node_loss = X * torch.log(node_pred) + (1 - X) * torch.log(1 - node_pred)
                node_loss = -node_loss.mean()
            elif cfg.loss_type == 'continuous_bernoulli':
                cb_dist = ContinuousBernoulli(probs=node_pred)
                node_loss = -cb_dist.log_prob(X).mean()
            elif cfg.loss_type =='mse':
                mse = (node_pred - X)**2
                node_loss = mse.mean()
        else:
            edge_loss = None
            node_loss = None


        return edge_loss, node_loss, loss_kld, mean_feature, features, z

    def calculate_feature_embedding(self, X, A, train = False):
        # Calculate the sequence of features.
        mean_feature, features, edge_cats = self.encoder(X, A)
        z_mean_post, z_std_post, z = self.sample_posterior(mean_feature) # q(|G
        return mean_feature, features, z, z_mean_post