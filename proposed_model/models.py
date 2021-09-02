import torch
import torch.nn.functional as F
from layers import GCNConv
from torch import nn


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

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device)

    def forward(self):
        x = self.ndata.weight
        # print(x)
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, self.adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, self.adj)

        x[0] = torch.zeros(x.shape[1])

        return x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device):
        super(MLP, self).__init__()

        self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_channels * 2, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))

        self.layers.append(torch.nn.Linear(hidden_channels, 1))

        self.dropout = dropout

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device)

    def embed_context(self, indices):
        embeddings = self.ndata(indices)
        context_emb = torch.sum(embeddings, dim=1)
        mask = indices > 0
        norm = torch.sum(mask, 1).view(-1, 1)
        return context_emb / norm

    def forward(self, indices, context=0):
        ing1 = self.ndata(indices[:, 0])
        ing2 = self.ndata(indices[:, 1])

        if context:
            context_emb = self.embed_context(indices[:, 2:])
            ing1 = ing1 + context_emb
            ing2 = ing2 + context_emb

        x = torch.cat((ing1, ing2), 1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x


class MLP_CAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device):
        super(MLP_CAT, self).__init__()
        self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_channels * 3, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels))

        self.layers.append(torch.nn.Linear(hidden_channels, 1))

        self.dropout = dropout

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device)

    def embed_context(self, indices):
        embeddings = self.ndata(indices)
        context_emb = torch.sum(embeddings, dim=1)
        mask = indices > 0
        norm = torch.sum(mask, 1).view(-1, 1)
        return context_emb / norm

    def forward(self, indices, context):
        ing1 = self.ndata(indices[:, 0])
        ing2 = self.ndata(indices[:, 1])

        context_emb = self.embed_context(indices[:, 2:])

        x = torch.cat((ing1, ing2, context_emb), 1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x


class DotP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device):
        super(DotP, self).__init__()

        self.ndata = nn.Embedding(adj.num_nodes(), in_channels)

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device)

    def forward(self):
        x = self.ndata.weight
        return x
