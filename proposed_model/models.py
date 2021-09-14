import torch
import torch.nn.functional as F
from layers import GCNConv
from torch import nn
import dgl.nn as dglnn
import pickle

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device):
        super(GAT, self).__init__()

        self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        self.layers = nn.ModuleList()

        num_heads = 2
        self.layers.append(dglnn.GATConv(
            in_feats=in_channels, out_feats=hidden_channels, num_heads=num_heads))
        for _ in range(num_layers - 1):
            self.layers.append(dglnn.GATConv(
            in_feats=hidden_channels*num_heads, out_feats=hidden_channels*num_heads, num_heads=1))

        self.dropout = dropout
        self.adj = adj
        self.adj.requires_grad = False

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device)

    def forward(self):
        x = self.ndata.weight
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(self.adj, x)
            x = x.view(-1, x.size(1) * x.size(2))
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](self.adj, x)
        x = x.squeeze()
        x[0] = torch.zeros(x.shape[1])
        return x

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

class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device):
        super(SAGE, self).__init__()

        self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(
            in_feats=in_channels, out_feats=hidden_channels, aggregator_type='mean'))
        for _ in range(num_layers - 1):
            self.layers.append(dglnn.SAGEConv(
            in_feats=hidden_channels, out_feats=hidden_channels, aggregator_type='mean'))

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
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(self.adj, x, self.adj.edata["w"])
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](self.adj, x, self.adj.edata["w"])

        x[0] = torch.zeros(x.shape[1])
        return x


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, node_count2id, node_id2name, mode="food_bert"):
        super(GIN, self).__init__()

        if mode == "random":
            self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        elif mode == "random_walk":
            from gensim.models import Word2Vec
            model = Word2Vec.load("/private/home/baharef/inversecooking2.0/proposed_model/node2vec/model_30_10.txt")
            embs = model.wv.load_word2vec_format("/private/home/baharef/inversecooking2.0/proposed_model/node2vec/embeddings_30_10.txt")
            self.ndata = torch.zeros(adj.num_nodes(), in_channels).to(device)
            for ind in range(adj.num_nodes()):
                self.ndata[ind] = torch.tensor(embs[str(ind)]).to(device)
        elif mode == "food_bert":
            self.ndata = pickle.load(open('/private/home/baharef/inversecooking2.0/proposed_model/node2vec/food_bert_emb.pkl', 'rb')).to(device)
          

            

        lin1 = torch.nn.Linear(in_channels, hidden_channels)
        lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GINConv(lin1, aggregator_type='mean', learn_eps=True))
        for _ in range(num_layers - 1):
            self.layers.append(dglnn.GINConv(lin2, aggregator_type='mean', learn_eps=True))

        self.dropout = dropout
        self.adj = adj
        self.adj.requires_grad = False

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device)

    def forward(self):
        # x = self.ndata.weight
        x = self.ndata
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(self.adj, x, self.adj.edata["w"])
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](self.adj, x, self.adj.edata["w"])

        x[0] = torch.zeros(x.shape[1])
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, mode="food_bert"):
        super(MLP, self).__init__()

        if mode == "random":
            self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        elif mode == "random_walk":
            from gensim.models import Word2Vec
            model = Word2Vec.load("/private/home/baharef/inversecooking2.0/proposed_model/node2vec/model_30_10.txt")
            embs = model.wv.load_word2vec_format("/private/home/baharef/inversecooking2.0/proposed_model/node2vec/embeddings_30_10.txt")
            self.ndata = torch.zeros(adj.num_nodes(), in_channels).to(device)
            for ind in range(adj.num_nodes()):
                self.ndata[ind] = torch.tensor(embs[str(ind)]).to(device)
        elif mode == "food_bert":
            self.ndata = pickle.load(open('/private/home/baharef/inversecooking2.0/proposed_model/node2vec/food_bert_emb.pkl', 'rb')).to(device)


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
        # ing1 = self.ndata(indices[:, 0])
        # ing2 = self.ndata(indices[:, 1])
        print(self.ndata)
        ing1 = self.ndata[indices[:, 0]]
        ing2 = self.ndata[indices[:, 1]]

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


class MLP_ATT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device):
        super(MLP_ATT, self).__init__()
        self.ndata = nn.Embedding(adj.num_nodes(), in_channels)


        self.transformer_layers = nn.ModuleList()
        self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=in_channels, nhead=5, batch_first=True))
        # self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=in_channels, nhead=5, batch_first=True))

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
        x = self.ndata(indices)
        mask = (indices > 0).long()
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, src_key_padding_mask=mask)
        norm = torch.sum(mask, 1).view(-1, 1)
        mask = mask.unsqueeze(2).repeat(1, 1, x.shape[2])
        x = torch.sum(x * mask, dim=1)/norm

        return x

    def embed_context_fast(self, embeddings, indices, nr):
        x = embeddings[0 : -1 : nr + 1]
        mask = (indices > 0).long()[0 : -1 : nr + 1]
        for i, layer in enumerate(self.transformer_layers):
            x = layer(x, src_key_padding_mask=mask)
            # x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        norm = torch.sum(mask, 1).view(-1, 1)
        mask = mask.unsqueeze(2).repeat(1, 1, x.shape[2])
        x = torch.sum(x * mask, dim=1)/norm

        return x.repeat(1, nr + 1).view(
            embeddings.shape[0], embeddings.shape[2]
        )

    def forward(self, indices, nr, mode="fast"):

        if mode == "slow":
            ing1 = self.ndata(indices[:, 0])
            ing2 = self.ndata(indices[:, 1])
            context_emb = self.embed_context(indices[:, 2:])
        else:
            embedding = self.ndata(indices)
            ing1 = embedding[:, 0]
            ing2 = embedding[:, 1]
            context_emb = self.embed_context_fast(embedding[:, 2:], indices[:, 2:], nr)

        x = torch.cat((ing1, ing2, context_emb), 1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)

        return x