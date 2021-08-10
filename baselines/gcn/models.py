import torch
from torch import nn
from baselines.gcn.layers import GCNConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device):
        super(GCN, self).__init__()

        self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.adj = adj
        print(self.adj)
        self.adj.requires_grad = False
        
        self.epoch = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False).to(device)
        self.mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False).to(device)

    def forward(self):
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(self.ndata.weight, self.adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, self.adj)
        return x

