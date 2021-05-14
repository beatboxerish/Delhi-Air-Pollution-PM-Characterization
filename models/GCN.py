# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import pandas as pd
import numpy as np
import os
import random
import tqdm
import torch
from torch_geometric.data import Data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(2, 4)
        self.batch1 = nn.BatchNorm1d(4)        
        self.conv2 = GCNConv(4, 3)
        self.batch2 = nn.BatchNorm1d(3)
        self.conv3 = nn.Linear(3, 4)
        self.batch3 = nn.BatchNorm1d(4)
        self.conv4 = nn.Linear(4, 3)
        self.conv5 = nn.Linear(3, 1)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.sigmoid(self.conv2(x, edge_index))
        self.saveLaplacianError(x, data)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return torch.squeeze(x)
    
    def saveLaplacianError(self, x, data):
        A = torch_geometric.utils.get_laplacian(data.edge_index)
        Amat = torch.zeros(data.x.shape[0], data.x.shape[0])
        for i in range(len(A[1])):
            Amat[A[0][0][i].item(),A[0][1][i].item()] = A[1][i].item()
        Amat = Amat.float()
        P = torch.mm(torch.mm(torch.transpose(x.float(),0,1),Amat.float()),x.float())
        self.llLoss = torch.sum(torch.diag(P))

def m_s_adjust(x):
    """
    Adjusts for mean and std for input/output normalization. Used in the next function.
    
    || INPUT ||
    x - dataset to standardize 
    
    || OUTPUT ||
    x - transformed dataset
    mx - mean of original dataset
    sx - std of original dataset
    """
    x = x.clone()
    mx = x.mean(0, keepdim=True)
    sx = x.std(0, unbiased=False, keepdim=True)
    x -= mx
    x /= sx
    return x, mx, sx

def normalize(train_tuple, test_tuple = None):
    """
    Normalize the train and/or test datasets
    
    || INPUT ||
    train_tuple - (train features, train output variable) 
    test_tuple - (test features, test output variable) 
    
    || OUTPUT ||
    outputs transformed train and test tuples (tuples as defined above) along with learnt mean and std. deviations
    """
    train_x, train_y = train_tuple
    train_x, train_y  = train_x.clone(), train_y.clone() 
    
    train_x, mx, sx = m_s_adjust(train_x)
    train_y, my, sy = m_s_adjust(train_y)
    train_tuple = (train_x, train_y)
    
    if test_tuple:
        test_x, test_y = test_tuple
        test_x, test_y  = test_x.clone(), test_y.clone() 
        test_x -= mx
        test_x /= sx
        test_y -= my
        test_y /= sy
        test_tuple = (test_x, test_y)
    return train_tuple, test_tuple, (mx, sx), (my, sy)

def make_edges(data_tuple, dist = 0.015):
    '''
    Make sure first two columns are lat and long and 3rd col is PM
    '''
    # df = df.values.copy()    
    X, y = data_tuple
    edgeList = []
    size = X.shape[0]
    # Adj_mat = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            if i!=j:
              ### correct this for haversine
                d = (X[i][0]-X[j][0])**2 + (X[i][1] - X[j][1])**2
                if(d<dist):
                    edgeList.append([i,j])
                    # Adj_mat[i][j] = 1
        ### why do we have self loops?
        edgeList.append([i,i])
            
    edges = np.array(edgeList)
    edges = edges.T
    return edges

def make_graph(data_tuple, trainMask, edges):
    X, y = data_tuple
    torch_X, torch_y, edge_mat, trainMask = torch.tensor(X), \
                                            torch.tensor(y, dtype = torch.float), \
                                            torch.tensor(edges, dtype = torch.long), \
                                            torch.tensor(trainMask, dtype = torch.long)

    graph = Data(x = torch_X, y = torch_y, edge_index = edge_mat,\
                 train_mask = trainMask)
    print("Graph Summary:", graph)
    return graph

def train(net, graph, optimizer):
    net.train()
    optimizer.zero_grad()
    preds = net(graph)
    loss = F.mse_loss(preds[graph.train_mask], graph.y[graph.train_mask])
    loss.backward()
    optimizer.step()
    return loss, preds

def train_GCN(graph, y_ms, num_epochs = 100, cuda = False):
    
    net = Net()
    net = net.float()
    if cuda:
        net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)

    epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
    for epoch in epochs_iter:
        
        loss, preds = train(net, graph, optimizer)
        
        if epoch%10 == 0:
            print("Epoch:", epoch+1)
            train_rmse, train_preds = evaluate_GCN(net, graph, y_ms)
            print("Train RMSE:", train_rmse.item())

def evaluate_GCN(net, graph, y_ms):
    net.eval()
    preds = net(graph)
    rmse = torch.sqrt(F.mse_loss(preds, graph.y))
    unnormalized_rmse = y_ms[1]*rmse
    return unnormalized_rmse, preds
