import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
import cfg
from copy import deepcopy
cfg = cfg.get_cfg()
from model import GCRN
device = torch.device(cfg.device)

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
            output_dim=2 * output_dim,
            hidden_units=hidden_units,
            hidden_activation=nn.LeakyReLU(0.2),
        ).apply(initialize_weight)


    def forward(self, x):
        x = self.net(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = F.softplus(std) + 1e-5
        return mean, std


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        device = torch.device(cfg.device)
        self.n_multi_head = params["n_multi_head"]
        self.Embedding = nn.Linear(params["num_of_process"],params["n_hidden"], bias=False).to(device)  # 그림 상에서 Encoder에 FF(feedforward)라고 써져있는 부분
        self.params = params
        self.k_hop = params["k_hop"]
        num_edge_cat = 3

        self.GraphEmbedding = GCRN(
                                   feature_size =  params["n_hidden"],
                                   graph_embedding_size= params["graph_embedding_size"],
                                   embedding_size =  params["n_hidden"],
                                   layers =  params["layers"],
                                   alpha =  params["alpha"],
                                   n_multi_head = params["n_multi_head"],
                                   num_edge_cat = num_edge_cat
                                   ).to(device)
        self.GraphEmbedding1 = GCRN(feature_size =  params["n_hidden"],
                                   graph_embedding_size= params["graph_embedding_size"],
                                   embedding_size =  params["n_hidden"],
                                    n_multi_head=params["n_multi_head"],
                                    layers =  params["layers"],
                                    alpha=params["alpha"],
                                    num_edge_cat = num_edge_cat).to(device)


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

        h = enc_h.mean(dim = 1) # 모든 node embedding에 대해서(element wise) 평균을 낸다.
        enc_h = enc_h[:, :-2]                 # dummy node(source, sink)는 제외한다.
        return h, enc_h, edge_cats





class GraphVAEDecoder(nn.Module):
    """
    GraphVAE decoder that takes a latent vector z and outputs a probabilistic fully-connected graph.
    The graph consists of:
    - Adjacency matrix A ∈ [0,1]^(k×k)
    - Edge attribute tensor E ∈ [0,1]^(k×k×de)
    - Node attribute matrix F ∈ [0,1]^(k×dn)

    Where:
    - k is the maximum number of nodes
    - de is the number of edge types
    - dn is the number of node types
    """

    def __init__(self, latent_dim, max_nodes=100, edge_types=3, node_types=4, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_nodes = max_nodes
        self.edge_types = edge_types
        self.node_types = node_types



        # Fully connected layers
        layers = []
        prev_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.fc_layers = nn.Sequential(*layers)

        # Output layers for adjacency matrix, edge attributes, and node attributes
        self.fc_edge = nn.Linear(hidden_dims[-1], max_nodes * max_nodes * edge_types)
        self.fc_node = nn.Linear(hidden_dims[-1], max_nodes * node_types)

    def forward(self, z):
        """
        Forward pass through the decoder.

        Args:
            z: Latent vectors of shape [batch_size, latent_dim]

        Returns:
            adj_prob: Adjacency matrix probabilities [batch_size, max_nodes, max_nodes]
            edge_prob: Edge type probabilities [batch_size, max_nodes, max_nodes, edge_types]
            node_prob: Node type probabilities [batch_size, max_nodes, node_types]
        """
        batch_size = z.size(0)

        # Compute feature representation
        h = self.fc_layers(z)


        # Compute edge type probabilities
        edge_logits = self.fc_edge(h).view(batch_size, self.max_nodes, self.max_nodes, self.edge_types)
        edge_prob = F.sigmoid(edge_logits)

        # Compute node type probabilities
        node_logits = self.fc_node(h).view(batch_size, self.max_nodes, self.node_types)
        node_prob = F.sigmoid(node_logits)
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
        self.z_posterior = Gaussian(
            params['n_hidden'],
            z_dim,
            hidden_units,
        )
        # p(G | z)
        self.decoder = GraphVAEDecoder(z_dim, edge_types=3, node_types=6, hidden_dims=[128, 256, 512]
        )
        #self.apply(initialize_weight)


    def sample_prior(self, x):
        z_mean, z_std = self.z_prior(x)
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


        loss_kld = calculate_kl_divergence(z_mean_post, z_std_post, z_mean_pri, z_std_pri).mean(dim=0).sum()

        if train == True:
            node_pred, edge_pred = self.decoder(z)
            node_pred = node_pred[:,:self.current_num_edges, :]
            edge_pred = edge_pred[:,:self.current_num_edges, :self.current_num_edges,:]

            edge_loss = edge_cats*torch.log(edge_pred)+(1-edge_cats)*torch.log(1-edge_pred)
            node_loss = X*torch.log(node_pred)+(1-X)*torch.log(1-node_pred)
            edge_loss = -edge_loss.sum([1,2]).mean()
            node_loss = -node_loss.sum(1).mean()
        else:
            edge_loss = None
            node_loss = None


        return edge_loss, node_loss, loss_kld, mean_feature, features, z