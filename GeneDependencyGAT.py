#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on [Date]
@author: [Your Name]
Description: GeneDependencyGAT Class
"""
import pandas as pd
import numpy as np
import json

import torch
from torch_geometric.utils import from_networkx
import networkx as nx
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F

from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch_geometric.data import Data, DataLoader

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool

import torch
from torch_geometric.data import Data, Batch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.nn import global_mean_pool

import torch
from torch_geometric.data import Data, Batch
import numpy as np
from torch_geometric.nn import GATConv, LayerNorm


def evaluate(model, loader):
    model.eval()
    preds_all, y_all = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index,data.batch)
            preds_all.append(out.cpu().numpy().flatten())
            y_all.append(data.y.cpu().numpy().flatten())

    y_true = np.concatenate(y_all)
    y_pred = np.concatenate(preds_all)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true.ravel(), y_pred.ravel())[0]

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'Pearson': pearson}

class GeneDependencyGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=2, dropout=0.2):
        super(GeneDependencyGAT, self).__init__()
    
        self.layer = nn.Linear(in_channels, 512)
        self.ln1 = nn.LayerNorm(512)
        self.dropout1 = nn.Dropout(0.1)#0.1,0.2
        
        #  GAT layers
        self.gat1 = GATConv(512, hidden_channels, heads=2, dropout=dropout,concat=True)
        self.norm1 = LayerNorm(hidden_channels * 2)

        self.gat2 = GATConv(hidden_channels * 2, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)

        # node level regression head
        self.lin = nn.Linear(hidden_channels * 2 , out_channels)

    def forward(self, x, edge_index, batch):
        
        x = self.layer(x)
        x = self.ln1(x)
        x = torch.relu(x)#relu,tanh
        
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)

        x = self.gat2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)


        x_global = global_mean_pool(x, batch)               # [num_graphs, hidden_dim]
        x_global = x_global[batch]                          # broadcast  [num_nodes, hidden_dim]

        x = torch.cat([x, x_global], dim=1)                 # [num_nodes, hidden*2]
        out = self.lin(x)
        return out    