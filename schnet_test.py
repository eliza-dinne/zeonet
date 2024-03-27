import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch_geometric.typing import OptTensor, SparseTensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.optim as optim 
from tqdm import tqdm
from numpy import save

from schnet import SchNet
from zeodata import create_graphs, unpack_batch, predict

import argparse
import os
import pickle

if __name__ == "__main__": #if file called, code below is executed
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--h", help="hidden channels and nr of filters",type=int, default=128)
    parser.add_argument("--i", help="nr of interactions",type=int, default=6)
    #more inputs can be added
    #possibly also the model
    args = parser.parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(DEVICE)

    d1 = {'hidden_channels': args.h, 'num_filters': args.h, 'num_interactions': args.i}

    #graph list creation
    graphs = []

    #'TON2', 'TONch', 'MEL', 'DDR', 'FAU', 'FAUch', 'ITW', 'MFI', 'MOR', 'RHO'
    for zeo in ['MFI']:
        _graphs = create_graphs(zeo, triplets=False)
        graphs.extend(_graphs)


    net = SchNet(**d1).to(DEVICE)

    test_size = 0.2
    n_train = int((1-test_size)*len(graphs))

    np.random.seed(0)
    np.random.shuffle(graphs)

    trainloader = DataLoader(graphs[:n_train], batch_size=32,shuffle=True)
    testloader = DataLoader(graphs[n_train:], batch_size=32,shuffle=True)

    epochs = 1

    optimizer = optim.AdamW(net.parameters())
    criterion = nn.HuberLoss()

    tr_loss = []
    te_loss = []


    for epoch in range(epochs):

        net.train()
        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader)): #move tqdm to the outer for loop

            optimizer.zero_grad() #clear gradients
            
            x, edge_index, edge_attr, y, batch = unpack_batch(data, DEVICE)
            out = net(x, edge_index, edge_attr, batch).squeeze() #perform forward pass
            loss = criterion(out, y) #compute loss based on training nodes

            loss.backward() #derive gradients
            optimizer.step() #update parameters based on gradients

            running_loss += loss.item()
            tr_loss.append(loss.item())
        

        print(f'Epoch {epoch+1} loss: {running_loss/(i+1)}')

        #test
        net.eval()
        running_loss_test = 0.0
        for i, data in enumerate(testloader):

            x, edge_index, edge_attr, y, batch = unpack_batch(data, DEVICE)
            with torch.no_grad():
                out = net(x, edge_index, edge_attr, batch).squeeze()
            loss = criterion(out, y)

            running_loss_test += loss.item()
            te_loss.append(loss.item())

        print(f'Epoch {epoch+1} test loss: {running_loss_test/(i+1)}')

    existing_folders = os.listdir('/workspaces/zeonet-bep/saved_results')
    existing_folders = [int(i) for i in existing_folders]

    if len(existing_folders) == 0:
        next_dir = 0
    else:
        next_dir = max(existing_folders) + 1

    os.makedirs(f'/Users/elizadinne/Desktop/uni/Physics_BEP/zeonet-bep/saved_results/{next_dir}/')
    torch.save(net.state_dict(), f'/Users/elizadinne/Desktop/uni/Physics_BEP/zeonet-bep/saved_results/{next_dir}/state_dict.py')
    with open(f'/Users/elizadinne/Desktop/uni/Physics_BEP/zeonet-bep/saved_results/{next_dir}/param_dict.pkl', 'wb') as f:
        pickle.dump(d1, f)


    #pred, true, zeos = predict(trainloader, net)
    #mae = np.abs(true - pred)
    #mse = ((true -pred)**2)

    #print(f'Overal: MAE: {mae.mean()} MSE: {mse.mean()}')
