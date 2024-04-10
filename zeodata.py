import torch
from torch import Tensor
import numpy as np
from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os

def get_tensor(file):
    return torch.tensor(np.load(file))

def get_atoms(file):

    with open(file) as f:
        lines = f.readlines()
    lines = [i.strip().split() for i in lines]
    lines = [i for i in lines if len(i)>1]


    at_pos = [i[1:5] for i in lines if i[1] in ['Si', 'Al']]
    atom = np.array([1 if i[0]=='Al' else 0 for i in at_pos])
    X = np.array([list(map(float, i[1:])) for i in at_pos])

    at_pos_O = [i[1:5] for i in lines if i[1] == 'O']
    X_o = np.array([list(map(float, i[1:])) for i in at_pos_O])
    return atom, X, X_o

def periodic_boundary(d):
    '''
    Applies periodic boundary conditions to the difference vector d (fractional coordinates)
    '''
    
    d = torch.where(d<-0.5, d+1, d)
    d = torch.where(d>0.5, d-1, d)
    
    return d

def get_transform_matrix(a, b, c, alpha, beta, gamma):
    """
    a, b, c: lattice vector lengths (angstroms)
    alpha, beta, gamma: lattice vector angles (degrees)

    Returns the transformation matrix from fractional to cartesian coordinates
    """
    # convert to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    
    zeta = (np.cos(alpha) - np.cos(gamma) * np.cos(beta))/np.sin(gamma)
    
    h = np.zeros((3,3))
    
    h[0,0] = a
    h[0,1] = b * np.cos(gamma)
    h[0,2] = c * np.cos(beta)

    h[1,1] = b * np.sin(gamma)
    h[1,2] = c * zeta

    h[2,2] = c * np.sqrt(1 - np.cos(beta)**2 - zeta**2)

    return h


def fractional_to_cartesian(X, h):
    '''
    X: (N, 3) tensor
    h: (3, 3) tensor
    '''
    return torch.matmul(X, h.T)

def cartesian_to_fractional(X, h):
    '''
    X: (N, 3) tensor
    h: (3, 3) tensor
    '''
    return torch.matmul(X, torch.inverse(h).T)


def get_distance(X1, X2, h): #NN
    '''
    Calculates pairwise distance between X1 and X2
    X1, X2: (N, 3) tensor (fractional coordinates)
    h: (3, 3) tensor

    Returns a (N,) tensor
    '''

    d_ij = X1 - X2
    d_ij = periodic_boundary(d_ij)
    d_ij = fractional_to_cartesian(d_ij, h)
    d_ij = torch.norm(d_ij, dim=1)
    return d_ij


def get_distance_matrix(X1, X2, h): #pre-process data
    '''
    Calculates pairwise distance matrix between X1 and X2
    X1: (N, 3) tensor (fractional coordinates)
    X2: (M, 3) tensor (fractional coordinates)
    h: (3, 3) tensor

    Returns a (N, M) tensor
    '''
    
    d_ij = X1.unsqueeze(1) - X2
    d_ij = periodic_boundary(d_ij)
    d_ij = fractional_to_cartesian(d_ij, h)
    d_ij = torch.norm(d_ij, dim=2)
    return d_ij

def get_edge_index(X, X_o, h):
    '''
    Gets edge index for the atoms based on T-O-T bonds
    X: (N, 3) tensor
    X_o: (N*2, 3) tensor
    h: (3, 3) tensor

    Returns: (N, N) tensor
    '''
    #nmpy array to torch
    X = torch.from_numpy(X)
    X_o = torch.from_numpy(X_o)
    h = torch.from_numpy(h)
    
    # calculate distance between X and X_o
    d_t_o = get_distance_matrix(X, X_o, h)
    idx_i, idx_j = d_t_o.argsort(dim=0)[:2,]

    # create edge index
    idx_1 = torch.cat([idx_i, idx_j], dim=0)
    idx_2 = torch.cat([idx_j, idx_i], dim=0)
    edge_index = torch.stack([idx_1, idx_2], dim=0)
    
    return edge_index

def get_triplets(edge_index):
    """
    Calculates i,j,k triplets for T-O-T-O-T bonds

    edge_index: (2, M) tensor
    """
    n = edge_index.max().item() + 1

    ind_i, ind_j = edge_index

    value = torch.arange(ind_j.size(0), device=ind_j.device)
    adj_t = SparseTensor(row=ind_i, col=ind_j, value=value,
                            sparse_sizes=(n,n))
    adj_t_row = adj_t[ind_j]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = ind_i.repeat_interleave(num_triplets)
    idx_j = ind_j.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return idx_i, idx_j, idx_k, idx_kj, idx_ji

def get_angles(X, h, idx_i_1, idx_j_1, idx_k_1):
    """
    Calculates the angle between T-O-T-O-T bonds

    X: (N, 3) tensor
    h: (3, 3) tensor
    idx_i_1: (M,) tensor
    idx_j_1: (M,) tensor
    idx_k_1: (M,) tensor

    Returns: (M,) tensor
    """

    d_ji_1 = X[idx_j_1] - X[idx_i_1]
    d_ji_1 = periodic_boundary(d_ji_1)

    d_kj_1 = X[idx_k_1] - X[idx_j_1]
    d_kj_1 = periodic_boundary(d_kj_1)

    d_kj_1 = fractional_to_cartesian(d_kj_1, h)
    d_ji_1 = fractional_to_cartesian(d_ji_1, h)

    a = (d_ji_1*d_kj_1).sum(dim=1)
    b = torch.cross(d_ji_1, d_kj_1).norm(dim=1)

    angle = torch.atan2(b, a)
    return angle

# We need to overwrite some methods from the Data class to ensure triplet batching is handled correctly
class ZeoData(Data):

    def __inc__(self, key, value, *args, **kwargs):
        
        if key in ['idx_kj_1', 'idx_ji_1']:
            return len(self.edge_attr)
        elif 'index' in key or 'idx' in key:
            return self.num_nodes
        else:
           return super().__inc__(key, value, *args, **kwargs)

def create_graphs(zeo : str = 'TON', triplets : bool = False):
    """
    Creates list of graphs for the given zeolite structure

    zeo: str (default: 'TON')
        Name of the zeolite structure
    triplets: bool (default: False)
        If True, returns graphs with triplets
    """

    graphs = []
    
    current_dir = os.getcwd()
    zeopath = f'{current_dir}/Data/{zeo}'
    
    # you might need to create these files before running
    X = get_tensor(f'{zeopath}/X.npy')
    atoms = get_tensor(f'{zeopath}/atoms.npy')
    adj = get_tensor(f'{zeopath}/adj.npy')
    l = get_tensor(f'{zeopath}/l.npy')
    angles = get_tensor(f'{zeopath}/angles.npy')
    y = get_tensor(f'{zeopath}/hoa.npy')


    h = torch.tensor(get_transform_matrix(*l, *angles))

    # edges, distances and angles always remain the same for a zeolite toplogy
    #idx_i, idx_j = torch.where(adj)

    edge_index = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            for k in range(adj[i, j].long().item()):
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index).T

    idx_i, idx_j = edge_index

    #edge_index = torch.stack([idx_i, idx_j])
    dists = get_distance(X[idx_i], X[idx_j], h)

    #if triplets:
    #    idx_i_1, idx_j_1, idx_k_1, idx_kj_1, idx_ji_1 = get_triplets(edge_index)
    #    angle = get_angles(X, h, idx_i_1, idx_j_1, idx_k_1)


    for i in range(atoms.shape[0]):

        if triplets:
            # we might not need all the indexing, check later
            data = ZeoData(x=atoms[i], edge_index=edge_index, edge_attr=dists.unsqueeze(1), 
                        idx_i_1=idx_i_1, idx_j_1=idx_j_1, idx_k_1=idx_k_1, idx_kj_1=idx_kj_1, idx_ji_1=idx_ji_1, angle=angle, y=y[i])
        else:
            data = Data(x=atoms[i], zeo=zeo, edge_index=edge_index, edge_attr=dists.unsqueeze(1), y=y[i])

        graphs.append(data)

    return graphs

def unpack_batch(batch, device : str = 'cpu'):
    '''
    Unpacks batch and moves data to device
 
    Parameters
    ----------
    batch : Batch
        batch of graph data
    device : str
        device to which data should be moved
 
    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        x, edge_index, edge_attr, orbit, orbit_weight, orbit_index, edge_orbit, edge_orbit_weight, edge_orbit_index, y, batch
    '''
    batch = batch.to(device)
    x = batch.x.float().unsqueeze(-1)
    edge_index = batch.edge_index
    edge_attr = batch.edge_attr.float()
    y = batch.y.float()
    batch = batch.batch
   
 
    return (x, edge_index, edge_attr, y, batch)


@torch.no_grad()
def predict(dataloader, model):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds = []
    trues = []
    #zeos = []
    for i, data in enumerate(dataloader):
        
        x, edge_index, edge_attr, y, batch = unpack_batch(data, DEVICE)
        out = model(x, edge_index, edge_attr, batch).squeeze()
        preds.append(out.cpu().numpy())
        trues.append(y.cpu().numpy())
        #zeos.extend(data.zeo)
    return np.concatenate(preds), np.concatenate(trues), None #np.array(zeos)