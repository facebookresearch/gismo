import torch
import torch.nn.functional as F
from layers import GCNConv
from torch import nn
import dgl.nn as dglnn
import pickle
import numpy as np

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
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, node_count2id, mode="flavorgraph"):
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
        elif mode == "flavorgraph":
            self.ndata = torch.zeros(adj.num_nodes(), in_channels).to(device)
            features_file = open("/private/home/baharef/inversecooking2.0/proposed_model/features/flavorgraph_emb.pkl", "rb")
            flavorgraph_embs = pickle.load(features_file)
            for count in node_count2id:
                id = node_count2id[count]
                try:
                    self.ndata[count] = torch.tensor(flavorgraph_embs[str(id)])
                except:
                    pass
                    
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


class GIN_MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, node_count2id, init_emb):
        super(GIN_MLP, self).__init__()

        if init_emb == "random":
            self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        elif init_emb == "food_bert":
            self.ndata = pickle.load(open('/private/home/baharef/inversecooking2.0/proposed_model/node2vec/food_bert_emb.pkl', 'rb')).to(device)
        elif init_emb == "flavorgraph":
            self.ndata = torch.zeros(adj.num_nodes(), in_channels).to(device)
            features_file = open("/private/home/baharef/inversecooking2.0/proposed_model/features/flavorgraph_emb.pkl", "rb")
            flavorgraph_embs = pickle.load(features_file)
            for count in node_count2id:
                id = node_count2id[count]
                try:
                    self.ndata[count] = torch.tensor(flavorgraph_embs[str(id)])
                except:
                    pass
        self.init_emb = init_emb
        self.with_titles = with_titles

        lin1 = torch.nn.Linear(in_channels, hidden_channels)
        lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin_layers = nn.ModuleList()
        self.gin_layers.append(dglnn.GINConv(lin1, aggregator_type='mean', learn_eps=True))
        for _ in range(num_layers - 1):
            self.gin_layers.append(dglnn.GINConv(lin2, aggregator_type='mean', learn_eps=True))

        self.dropout = dropout
        self.adj = adj
        self.adj.requires_grad = False

        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(torch.nn.Linear(hidden_channels * 2, hidden_channels))
        for _ in range(num_layers - 1):
            self.mlp_layers.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
        self.mlp_layers.append(torch.nn.Linear(hidden_channels//2, 1))
        
        if self.with_titles:
            embs = pickle.load(open('/private/home/baharef/inversecooking2.0/preprocessed_data/title_embeddings.pkl', 'rb'))
            ids = pickle.load(open('/private/home/baharef/inversecooking2.0/preprocessed_data/title_recipe_ids.pkl', 'rb'))
            emb_dim = len(embs[0])
            self.title_embeddings = torch.zeros((len(embs),emb_dim)).to(device)
            for id_ in recipe_id2counter:
                counter = recipe_id2counter[id_]
                self.title_embeddings[counter] = embs[ids.index(id_)]
            self.title_project_layer = torch.nn.Linear(emb_dim, hidden_channels)
            self.mlp_layers_context = nn.ModuleList()
            self.mlp_layers_context.append(torch.nn.Linear(hidden_channels * 4, hidden_channels))
            for _ in range(num_layers - 1):
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
            self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))
        else:
            self.mlp_layers_context = nn.ModuleList()
            self.mlp_layers_context.append(torch.nn.Linear(hidden_channels * 3, hidden_channels))
            for _ in range(num_layers - 1):
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
            self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))
        
        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device)

    def embed_context(self, indices, x):
        embeddings = torch.index_select(x, 0, indices.reshape(-1)).reshape(indices.shape[0], indices.shape[1], x.shape[1])
        context_emb = torch.sum(embeddings, dim=1)
        mask = indices > 0
        norm = torch.sum(mask, 1).view(-1, 1)
        norm[norm == 0] = 1
        return context_emb / norm

    def embed_title(self, indices):
        return self.title_project_layer(torch.index_select(self.title_embeddings, 0, indices.reshape(-1)))

    def forward(self, indices, context):
        if self.init_emb == "random":
            x = self.ndata.weight
        else:
            x = self.ndata
        for i, conv in enumerate(self.gin_layers[:-1]):
            x = conv(self.adj, x, self.adj.edata["w"])
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gin_layers[-1](self.adj, x, self.adj.edata["w"])

        x[0] = torch.zeros(x.shape[1], dtype=torch.float).cuda()

        ing1 = x[indices[:, 0]]
        ing2 = x[indices[:, 1]]

        if context:
            context_emb = self.embed_context(indices[:, 3:], x)
            if self.with_titles:
                title_emb = self.embed_title(indices[:, 2])
                y = torch.cat((ing1, ing2, context_emb, title_emb), 1)
            else:
                y = torch.cat((ing1, ing2, context_emb), 1)
            layers = self.mlp_layers_context
        else:
            y = torch.cat((ing1, ing2), 1)
            layers = self.mlp_layers

        for i, layer in enumerate(layers[:-1]):
            y = layer(y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = layers[-1](y)

        return y

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, node_count2id, init_emb):
        super(MLP, self).__init__()

        self.init_emb = init_emb
        if self.init_emb == "random":
            self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        elif self.init_emb == "random_walk":
            from gensim.models import Word2Vec
            model = Word2Vec.load("/private/home/baharef/inversecooking2.0/proposed_model/node2vec/model_30_10.txt")
            embs = model.wv.load_word2vec_format("/private/home/baharef/inversecooking2.0/proposed_model/node2vec/embeddings_30_10.txt")
            self.ndata = torch.zeros(adj.num_nodes(), in_channels).to(device)
            for ind in range(adj.num_nodes()):
                self.ndata[ind] = torch.tensor(embs[str(ind)]).to(device)
        elif self.init_emb == "food_bert":
            self.ndata = pickle.load(open('/private/home/baharef/inversecooking2.0/proposed_model/node2vec/food_bert_emb.pkl', 'rb')).to(device)


        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_channels * 2, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels//2))

        self.layers.append(torch.nn.Linear(hidden_channels//2, 1))

        self.dropout = dropout

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device)

    # def embed_context(self, indices):
    #     embeddings = self.ndata(indices)
    #     context_emb = torch.sum(embeddings, dim=1)
    #     mask = indices > 0
    #     norm = torch.sum(mask, 1).view(-1, 1)
    #     return context_emb / norm

    def forward(self, indices, context=0):
        if self.init_emb == "random":
            ing1 = self.ndata(indices[:, 0])
            ing2 = self.ndata(indices[:, 1])
        else:
            ing1 = self.ndata[indices[:, 0]]
            ing2 = self.ndata[indices[:, 1]]

        # if context:
        #     context_emb = self.embed_context(indices[:, 3:])
        #     ing1 = ing1 + context_emb
        #     ing2 = ing2 + context_emb

        x = torch.cat((ing1, ing2), 1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x


class MLP_CAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, node_count2id, init_emb):
        super(MLP_CAT, self).__init__()
        self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        self.layers = nn.ModuleList()
        self.layers.append(torch.nn.Linear(in_channels * 3, hidden_channels))
        for _ in range(num_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_channels, hidden_channels//2))

        self.layers.append(torch.nn.Linear(hidden_channels//2, 1))
        
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

    def forward(self, indices):
        ing1 = self.ndata(indices[:, 0])
        ing2 = self.ndata(indices[:, 1])

        context_emb = self.embed_context(indices[:, 3:])

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
        self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=in_channels, nhead=5, dim_feedforward=2*in_channels , batch_first=True))
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
            context_emb = self.embed_context(indices[:, 3:])
        else:
            embedding = self.ndata(indices)
            ing1 = embedding[:, 0]
            ing2 = embedding[:, 1]
            context_emb = self.embed_context_fast(embedding[:, 3:], indices[:, 3:], nr)

        x = torch.cat((ing1, ing2, context_emb), 1)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)

        return x



class LT(nn.Module):
    def __init__(self, train_table):
        super(LT, self).__init__()
        self.train_table = train_table

    def forward(self, indices):
        x = self.train_table[indices[:, 0], indices[:, 1]]
        return x
