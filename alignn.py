import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_geometric.utils import scatter
from torch_geometric.nn.aggr import MeanAggregation



class RBFExpansion(nn.Module):
    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale = None,
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(self.vmin, self.vmax, self.bins)
        )

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return torch.exp(
            -self.gamma * (distance - self.centers) ** 2
        )

class EdgeGatedGraphConv(nn.Module):
    
    def __init__(self, input_features: int, output_features: int, residual: bool = True):

        super().__init__()
        self.residual = residual
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)
        
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)
    
    def forward(self, node_feats, edge_feats, edge_index):
        idx_i, idx_j = edge_index
        e_src = self.src_gate(node_feats)[idx_i]
        e_dst = self.dst_gate(node_feats)[idx_j]

        y = e_src + e_dst + self.edge_gate(edge_feats)
        sigma = torch.sigmoid(y)
        bh = self.dst_update(node_feats)[idx_j]
        m = bh*sigma

        sum_sigma_h = scatter(m, idx_i, 0)

        sum_sigma = scatter(sigma, idx_i, 0)

        h = sum_sigma_h/(sum_sigma+1e-6)

        x = self.src_update(node_feats) + h

        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(y))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y

class ALIGNNConv(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(self, x, y, z, edge_index, edge_index_triplets):

        m, z = self.edge_update(y, z, edge_index_triplets)
        x, y = self.node_update(x, m, edge_index)

        return x, y, z

class MLPLayer(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()
        self.layer = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        return F.silu(self.bn(self.layer(x)))

class ALIGNN(nn.Module):

    def __init__(self, embedding_features=64, triplet_input_features=40, hidden_features=256, output_features=1, mx_d=8, centers=80, a_layers=4, g_layers=4):
        super().__init__()

        self.atom_embedding = nn.Sequential(MLPLayer(1, hidden_features))
        self.edge_embedding = nn.Sequential(RBFExpansion(vmin=0, vmax=8.0, bins=centers),
                                            MLPLayer(centers, embedding_features),
                                            MLPLayer(embedding_features, hidden_features))
        self.angle_embedding = nn.Sequential(RBFExpansion(vmin=-1, vmax=1.0, bins=triplet_input_features),
                                             MLPLayer(triplet_input_features, embedding_features), 
                                             MLPLayer(embedding_features, hidden_features))

        self.alignn_layers = nn.ModuleList([ALIGNNConv(hidden_features, hidden_features) for _ in range(a_layers)])
        self.gcn_layers = nn.ModuleList([EdgeGatedGraphConv(hidden_features, hidden_features) for _ in range(g_layers)])
        #alignn layers can be set to 0 to emulate cgcn

        self.readout = MeanAggregation()
        self.out = nn.Linear(hidden_features, output_features)
    

    def forward(self, x, edge_index, edge_index_triplets, dist, angle, batch):
        #print("x:",x.shape, "edge_index:", edge_index.shape, "edge_index_triplets", edge_index_triplets.shape, "dist:", dist.shape, "angle", angle.shape)  
        x = self.atom_embedding(x)
        y = self.edge_embedding(dist)
        z = self.angle_embedding(angle)

        for layer in self.alignn_layers:
            x, y, z = layer(x, y, z, edge_index, edge_index_triplets)

        for layer in self.gcn_layers:
            x, y = layer(x, y, edge_index)

        h = self.readout(x, batch)
        out = self.out(h)

        return out