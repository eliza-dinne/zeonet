import torch
import torch.optim as optim 
from zeodata_alignn import ZeoData, create_graphs, unpack_batch, get_tensor, get_transform_matrix, get_distance, get_atoms, periodic_boundary, fractional_to_cartesian, cartesian_to_fractional, get_distance_matrix, get_edge_index
from schnet import SchNet, InteractionBlock, CFConv, GaussianSmearing, ShiftedSoftplus
from alignn import ALIGNN
import os
import pickle
import numpy as np
from torch_geometric.loader import DataLoader

if __name__ == "__main__":

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(DEVICE)

    graphs1 = [] #training
    graphs2 = [] #testing

    #'TON', 'TON2', 'TON3', 'TON4', 'TONch', 'MEL', 'MELch', 'DDR', 'DDRch1', 'DDRch2', 'FAU', 'FAUch', 'ITW', 'MFI', 'MOR', 'RHO'
    for zeo in ['TON', 'TON2', 'TON3', 'TON4', 'TONch', 'MEL', 'MELch', 'DDR', 'DDRch1', 'DDRch2', 'FAU', 'FAUch', 'ITW', 'MFI', 'MOR']:
        _graphs1 = create_graphs(zeo, triplets=True)
        graphs1.extend(_graphs1)

    for zeo in ['RHO']:
        _graphs2 = create_graphs(zeo, triplets=True)
        graphs2.extend(_graphs2)

    print('RHO test')
    
    #test_size = 0.2
    #n_train = int((1-test_size)*len(graphs))

    #np.random.seed(0)
    #np.random.shuffle(graphs)

    trainloader = DataLoader(graphs1, batch_size=32,shuffle=True)
    testloader = DataLoader(graphs2, batch_size=32,shuffle=True)

    def predict(dataloader, model):
        model.eval()
        preds = []
        trues = []
        zeos = []
        for i, data in enumerate(dataloader):
    
            x, edge_index, edge_index_triplets, dist, angle, y, batch = unpack_batch(data, DEVICE)
            out = model(x, edge_index, edge_index_triplets, dist, angle, batch).squeeze()
            #x, edge_index, edge_attr, y, batch = unpack_batch(data, DEVICE)
            #out = model(x, edge_index, edge_attr, batch).squeeze()
            preds.append(out.cpu().numpy())
            trues.append(y.cpu().numpy())
            zeos.extend(data.zeo)
        return np.concatenate(preds), np.concatenate(trues), np.array(zeos)

    current_dir = os.getcwd()

    #for i in range(0,12):
    with open(f'{current_dir}/saved_results/0/param_dict.pkl', 'rb') as param_dict:
        d = pickle.load(param_dict)

    net = ALIGNN(**d).to(DEVICE)

    net.load_state_dict(torch.load(f'{current_dir}/saved_results/0/state_dict.py'))
    print(d)

    pred, true, zeos = predict(testloader, net)
    # calculate the MAE and MSE
    mae = np.abs(true - pred)
    mse = ((true -pred)**2)

    print(f'Overall: MAE: {mae.mean()} MSE: {mse.mean()}')