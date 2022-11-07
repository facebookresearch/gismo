# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
import dgl.nn as dglnn
import pickle


class GIN_MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, with_set, node_count2id, init_emb, context_emb_mode, pool, cfg):
        super(GIN_MLP, self).__init__()

        if init_emb == "random":
            self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        elif init_emb == "food_bert":
            self.ndata = pickle.load(open(cfg.init_emb_paths.food_bert, 'rb')).to(device)
        elif init_emb == "food_bert2":
            self.ndata1 = nn.Embedding(adj.num_nodes(), in_channels)
            nn.init.uniform_(self.ndata1.weight.data, 0.0, 0.001)
            self.ndata2 = pickle.load(open(cfg.init_emb_paths.food_bert, 'rb')).to(device)
        elif init_emb == "flavorgraph":
            self.ndata = torch.zeros(adj.num_nodes(), in_channels).to(device)
            features_file = open(cfg.init_emb_paths.flavorgraph, "rb")
            flavorgraph_embs = pickle.load(features_file)
            for count in node_count2id:
                id = node_count2id[count]
                try:
                    self.ndata[count] = torch.tensor(flavorgraph_embs[str(id)])
                except:
                    pass
        elif init_emb == "flavorgraph2":
            self.ndata1 = nn.Embedding(adj.num_nodes(), in_channels)
            self.ndata2 = torch.zeros(adj.num_nodes(), in_channels).to(device)
            features_file = open(cfg.init_emb_paths.flavorgraph, "rb")
            flavorgraph_embs = pickle.load(features_file)
            for count in node_count2id:
                id = node_count2id[count]
                try:
                    self.ndata2[count] = torch.tensor(flavorgraph_embs[str(id)])
                except:
                    pass
            
        self.init_emb = init_emb
        self.with_titles = with_titles
        self.with_set = with_set

        lin1 = torch.nn.Linear(in_channels, hidden_channels)
        lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin_layers = nn.ModuleList()
        self.gin_layers.append(dglnn.GINConv(lin1, aggregator_type='mean', learn_eps=True))
        for _ in range(num_layers - 1):
            self.gin_layers.append(dglnn.GINConv(lin2, aggregator_type='mean', learn_eps=True))

        self.dropout = dropout
        self.adj = adj
        self.adj.requires_grad = False

        if self.with_titles:
            embs = pickle.load(open(cfg.titles_paths.embedding, 'rb'))
            ids = pickle.load(open(cfg.titles_paths.ids, 'rb'))
            emb_dim = len(embs[0])
            self.title_embeddings = torch.zeros((len(embs),emb_dim)).to(device)
            for id_ in recipe_id2counter:
                counter = recipe_id2counter[id_]
                self.title_embeddings[counter] = embs[ids.index(id_)]
            self.title_project_layer = torch.nn.Linear(emb_dim, hidden_channels)

            if self.with_set:
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
        else:
            if self.with_set:
                self.mlp_layers_context = nn.ModuleList()
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels * 3, hidden_channels))
                for _ in range(num_layers - 1):
                    self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))
            else:
                self.mlp_layers_context = nn.ModuleList()
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels * 2, hidden_channels))
                for _ in range(num_layers - 1):
                    self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))

        self.context_emb_mode = context_emb_mode
        if self.context_emb_mode == "transformer":
            self.transformer_layers = nn.ModuleList()
            self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=5, dim_feedforward=2*hidden_channels , batch_first=True))
        elif self.context_emb_mode == "transformer2":
            self.cls_token = nn.Embedding(1, hidden_channels)
            self.transformer_layers = nn.ModuleList()
            self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=5, dim_feedforward=2*hidden_channels , batch_first=True))
            self.transformer_layers.append(nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=5, dim_feedforward=2*hidden_channels , batch_first=True))

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device)


        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(torch.nn.Linear(hidden_channels * 2, hidden_channels))
        for _ in range(num_layers - 1):
            self.mlp_layers.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
        self.mlp_layers.append(torch.nn.Linear(hidden_channels//2, 1))
        

    def embed_context(self, indices, x):
        if self.context_emb_mode == "avg":
            embeddings = torch.index_select(x, 0, indices.reshape(-1)).reshape(indices.shape[0], indices.shape[1], x.shape[1])
            context_emb = torch.sum(embeddings, dim=1)
            mask = indices > 0
            norm = torch.sum(mask, 1).view(-1, 1)
            norm[norm == 0] = 1
            return context_emb / norm
        elif self.context_emb_mode == "transformer":
            context_emb = torch.index_select(x, 0, indices.reshape(-1)).reshape(indices.shape[0], indices.shape[1], x.shape[1])
            mask = (indices == 0)
            for i, layer in enumerate(self.transformer_layers):
                context_emb = layer(context_emb, src_key_padding_mask=mask)
            mask2 = (indices > 0)
            norm = torch.sum(mask2, 1).view(-1, 1)
            mask2 = mask2.unsqueeze(2).repeat(1, 1, context_emb.shape[2])
            context_emb = torch.sum(context_emb * mask2, dim=1)/norm
            return context_emb
        elif self.context_emb_mode == "transformer2":
            context_emb = torch.index_select(x, 0, indices.reshape(-1)).reshape(indices.shape[0], indices.shape[1], x.shape[1])
            cls_emb = self.cls_token(torch.tensor([0]).cuda()).repeat(context_emb.shape[0], 1).unsqueeze(1)
            context_emb = torch.cat((cls_emb, context_emb), 1)
            mask_ind = (indices == 0)
            mask_cls = torch.full((context_emb.shape[0], 1), False).cuda()
            mask = torch.cat((mask_cls, mask_ind), 1)
            for i, layer in enumerate(self.transformer_layers):
                context_emb = layer(context_emb, src_key_padding_mask=mask)
            return context_emb[:, 0, :]

    def embed_context_fast(self, indices, embeddings, nr):
        if self.context_emb_mode == "avg":
            new_indices = indices[:-1:nr+1]
            embeddings = torch.index_select(embeddings, 0, new_indices.reshape(-1)).reshape(new_indices.shape[0], new_indices.shape[1], embeddings.shape[1])
            context_emb = torch.sum(embeddings, dim=1)
            mask = new_indices > 0
            norm = torch.sum(mask, 1).view(-1, 1)
            norm[norm == 0] = 1
            x =  context_emb / norm
            return x.repeat(1, nr + 1).view(indices.shape[0], embeddings.shape[2])
            
        elif self.context_emb_mode == "transformer":
            new_indices = indices[:-1:nr+1]
            context_emb = torch.index_select(embeddings, 0, new_indices.reshape(-1)).reshape(new_indices.shape[0], new_indices.shape[1], embeddings.shape[1])
            mask = (new_indices == 0)
            for i, layer in enumerate(self.transformer_layers):
                context_emb = layer(context_emb, src_key_padding_mask=mask)
            mask2 = (new_indices > 0)
            norm = torch.sum(mask2, 1).view(-1, 1)
            mask2 = mask2.unsqueeze(2).repeat(1, 1, context_emb.shape[2])
            context_emb = torch.sum(context_emb * mask2, dim=1)/norm
            y = context_emb.repeat(1, nr + 1).view(indices.shape[0], context_emb.shape[1])
            return y

        elif self.context_emb_mode == "transformer2":
            new_indices = indices[:-1:nr+1]
            context_emb = torch.index_select(embeddings, 0, new_indices.reshape(-1)).reshape(new_indices.shape[0], new_indices.shape[1], embeddings.shape[1])
            cls_emb = self.cls_token(torch.tensor([0]).cuda()).repeat(context_emb.shape[0], 1).unsqueeze(1)
            context_emb = torch.cat((cls_emb, context_emb), 1)
            mask_ind = (new_indices == 0)
            mask_cls = torch.full((context_emb.shape[0], 1), False).cuda()
            mask = torch.cat((mask_cls, mask_ind), 1)
            for i, layer in enumerate(self.transformer_layers):
                context_emb = layer(context_emb, src_key_padding_mask=mask)
            y = context_emb[:, 0, :].repeat(1, nr + 1).view(indices.shape[0], context_emb.shape[2])
            return y

    def embed_title(self, indices):
        return self.title_project_layer(torch.index_select(self.title_embeddings, 0, indices.reshape(-1)))

    def forward(self, indices, context, nr):
        if self.init_emb == "random":
            x = self.ndata.weight
        elif self.init_emb == "flavorgraph2" or self.init_emb == "food_bert2":
            x = self.ndata1.weight + self.ndata2
        else:
            x = self.ndata

        for i, conv in enumerate(self.gin_layers[:-1]):
            x = conv(self.adj, x, self.adj.edata["w"])
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gin_layers[-1](self.adj, x, self.adj.edata["w"])

        x[0] = torch.zeros(x.shape[1], dtype=torch.float).cuda()

        ing1 = torch.index_select(x, 0, indices[:, 0].view(-1))
        ing2 = torch.index_select(x, 0, indices[:, 1].view(-1))


        if self.with_titles:
            title_emb = self.embed_title(indices[:, 2])
            if self.with_set:
                set_emb = self.embed_context_fast(indices[:, 3:], x, nr)
                y = torch.cat((ing1, ing2, set_emb, title_emb), 1)
            else:
                y = torch.cat((ing1, ing2, title_emb), 1)
        else:
            if self.with_set:
                set_emb = self.embed_context_fast(indices[:, 3:], x, nr)
                y = torch.cat((ing1, ing2, set_emb), 1)
            else:
                y = torch.cat((ing1, ing2), 1)

        for i, layer in enumerate(self.mlp_layers_context[:-1]):
            y = layer(y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.mlp_layers_context[-1](y)

        return y

class GIN_MLP2(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, node_count2id, init_emb, context_emb_mode, pool, cfg):
        super(GIN_MLP2, self).__init__()

        if init_emb == "random":
            self.ndata = nn.Embedding(adj.num_nodes(), in_channels)
        elif init_emb == "food_bert":
            self.ndata = pickle.load(open(cfg.init_emb_paths.food_bert, 'rb')).to(device)
        elif init_emb == "flavorgraph":
            self.ndata = torch.zeros(adj.num_nodes(), in_channels).to(device)
            features_file = open(cfg.init_emb_paths.flavorgraph, "rb")
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
        
        
        self.mlp_layers_context = nn.ModuleList()
        self.mlp_layers_context.append(torch.nn.Linear(hidden_channels * 3, hidden_channels))
        for _ in range(num_layers - 1):
            self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
        self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))
        

        self.context_emb_mode = context_emb_mode
        self.pool = pool

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device)

    def forward(self, indices, context, nr):
    
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

        if context:
            ings = indices[:, :2]
            new_ings = ings.unsqueeze(1).repeat(1, indices.shape[1] - 3, 1)
            context = indices[:, 3:]
            new_indices = torch.cat((new_ings, context.reshape(new_ings.shape[0], new_ings.shape[1], 1)), 2)

            ing1 = torch.index_select(x, 0, new_indices[:, :, 0].view(-1))
            ing2 = torch.index_select(x, 0, new_indices[:, :, 1].view(-1))
            ing3 = torch.index_select(x, 0, new_indices[:, :, 2].view(-1))
            
            y = torch.cat((ing1, ing2, ing3), 1)
            layers = self.mlp_layers_context

        for i, layer in enumerate(layers[:-1]):
            y = layer(y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = layers[-1](y)
        
        y = y.reshape(new_indices.shape[0], new_indices.shape[1])

        if self.pool == "avg":
            mask = indices[:, 3:] > 0
            norm = torch.sum(mask, 1)
            sims = torch.sum(y*mask, dim=1)/norm
        elif self.pool == "min":
            mask = (indices[:, 3:]==0)*100000
            sims = torch.min(y+mask, dim=1).values
        elif self.pool == "max":
            mask = (indices[:, 3:]==0)*100000
            sims = torch.max(y-mask, dim=1).values
        else:
            print("Pooling method is not defined!")
            exit()
            
        sims = sims.reshape(-1, 1)
        return sims

class MLP_CAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, with_set, node_count2id, init_emb, context_emb_mode, pool, cfg):
        super(MLP_CAT, self).__init__()
        self.ndata = nn.Embedding(adj.num_nodes(), in_channels)

        self.dropout = dropout
        self.with_titles = with_titles
        self.with_set = with_set

        if self.with_titles:
            embs = pickle.load(open(cfg.titles_paths.embedding, 'rb'))
            ids = pickle.load(open(cfg.titles_paths.ids, 'rb'))
            emb_dim = len(embs[0])
            self.title_embeddings = torch.zeros((len(embs),emb_dim)).to(device)
            for id_ in recipe_id2counter:
                counter = recipe_id2counter[id_]
                self.title_embeddings[counter] = embs[ids.index(id_)]
            self.title_project_layer = torch.nn.Linear(emb_dim, in_channels)

            if self.with_set:
                self.mlp_layers_context = nn.ModuleList()
                self.mlp_layers_context.append(torch.nn.Linear(in_channels * 4, hidden_channels))
                for _ in range(num_layers - 1):
                    self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))
            else:
                self.mlp_layers_context = nn.ModuleList()
                self.mlp_layers_context.append(torch.nn.Linear(in_channels * 3, hidden_channels))
                for _ in range(num_layers - 1):
                    self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))
        else:
            if self.with_set:
                self.mlp_layers_context = nn.ModuleList()
                self.mlp_layers_context.append(torch.nn.Linear(in_channels * 3, hidden_channels))
                for _ in range(num_layers - 1):
                    self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))
            else:
                self.mlp_layers_context = nn.ModuleList()
                self.mlp_layers_context.append(torch.nn.Linear(in_channels * 2, hidden_channels))
                for _ in range(num_layers - 1):
                    self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))

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
        norm[norm == 0] = 1
        return context_emb / norm

    def embed_title(self, indices):
        return self.title_project_layer(torch.index_select(self.title_embeddings, 0, indices.reshape(-1)))

    def forward(self, indices):
        ing1 = self.ndata(indices[:, 0])
        ing2 = self.ndata(indices[:, 1])

        if self.with_titles:
            title_emb = self.embed_title(indices[:, 2])
            if self.with_set:
                set_emb = self.embed_context(indices[:, 3:])
                x = torch.cat((ing1, ing2, set_emb, title_emb), 1)
            else:
                x = torch.cat((ing1, ing2, title_emb), 1)
        else:
            if self.with_set:
                set_emb = self.embed_context(indices[:, 3:])
                x = torch.cat((ing1, ing2, set_emb), 1)
            else:
                x = torch.cat((ing1, ing2), 1)
        
        for i, layer in enumerate(self.mlp_layers_context[:-1]):
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.mlp_layers_context[-1](x)

        return x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, with_set, node_count2id, init_emb, context_emb_mode, pool, cfg):
        super(MLP, self).__init__()
        self.with_titles = with_titles
        self.with_set = with_set

        self.dropout = dropout
        self.adj = adj
        self.adj.requires_grad = False

        if self.with_titles:
            embs = pickle.load(open(cfg.titles_paths.embedding, 'rb'))
            ids = pickle.load(open(cfg.titles_paths.ids, 'rb'))
            emb_dim = len(embs[0])
            self.title_embeddings = torch.zeros((len(embs),emb_dim)).to(device)
            for id_ in recipe_id2counter:
                counter = recipe_id2counter[id_]
                self.title_embeddings[counter] = embs[ids.index(id_)]
            self.title_project_layer = torch.nn.Linear(emb_dim, hidden_channels)

            if self.with_set:
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
        else:
            if self.with_set:
                self.mlp_layers_context = nn.ModuleList()
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels * 3, hidden_channels))
                for _ in range(num_layers - 1):
                    self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))
            else:
                self.mlp_layers_context = nn.ModuleList()
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels * 2, hidden_channels))
                for _ in range(num_layers - 1):
                    self.mlp_layers_context.append(torch.nn.Linear(hidden_channels, hidden_channels//2))
                self.mlp_layers_context.append(torch.nn.Linear(hidden_channels//2, 1))

        self.context_emb_mode = context_emb_mode

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device) 

    def embed_context(self, indices, embeddings, nr):
        new_indices = indices[:-1:nr+1]
        embeddings = torch.index_select(embeddings, 0, new_indices.reshape(-1)).reshape(new_indices.shape[0], new_indices.shape[1], embeddings.shape[1])
        context_emb = torch.sum(embeddings, dim=1)
        mask = new_indices > 0
        norm = torch.sum(mask, 1).view(-1, 1)
        norm[norm == 0] = 1
        x =  context_emb / norm
        return x.repeat(1, nr + 1).view(indices.shape[0], embeddings.shape[2])

    def embed_title(self, indices):
        return self.title_project_layer(torch.index_select(self.title_embeddings, 0, indices.reshape(-1)))

    def forward(self, indices, context, nr, x):
        ing1 = torch.index_select(x, 0, indices[:, 0].view(-1))
        ing2 = torch.index_select(x, 0, indices[:, 1].view(-1))

        if self.with_titles:
            title_emb = self.embed_title(indices[:, 2])
            if self.with_set:
                set_emb = self.embed_context(indices[:, 3:], x, nr)
                y = torch.cat((ing1, ing2, set_emb, title_emb), 1)
            else:
                y = torch.cat((ing1, ing2, title_emb), 1)
        else:
            if self.with_set:
                set_emb = self.embed_context(indices[:, 3:], x, nr)
                y = torch.cat((ing1, ing2, set_emb), 1)
            else:
                y = torch.cat((ing1, ing2), 1)

        for i, layer in enumerate(self.mlp_layers_context[:-1]):
            y = layer(y)
            y = F.relu(y)
            y = F.dropout(y, p=self.dropout, training=self.training)
        y = self.mlp_layers_context[-1](y)

        return y

class GIST(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, with_set, node_count2id, init_emb, context_emb_mode, pool, cfg):
        super(GIST, self).__init__()
        self.gin = GIN(in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, with_set, node_count2id, init_emb, context_emb_mode, pool)
        self.mlp = MLP(in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, with_set, node_count2id, init_emb, context_emb_mode, pool)

        self.epoch = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.int32), requires_grad=False
        ).to(device)
        self.mrr = torch.nn.Parameter(
            torch.tensor(0, dtype=torch.float64), requires_grad=False
        ).to(device) 

    def forward(self, indices, context, nr):
        x = self.gin(indices, context, nr)
        return self.mlp(indices, context, nr, x)


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout, adj, device, recipe_id2counter, with_titles, with_set, node_count2id, init_emb, context_emb_mode, pool, cfg):
        super(GIN, self).__init__()

        self.ndata = nn.Embedding(adj.num_nodes(), in_channels)

        lin1 = torch.nn.Linear(in_channels, hidden_channels)
        lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GINConv(lin1, aggregator_type='mean', learn_eps=True))
        for _ in range(num_layers - 1):
            self.layers.append(dglnn.GINConv(lin2, aggregator_type='mean', learn_eps=True))


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

class LTFreq(nn.Module):
    def __init__(self, train_table):
        super(LTFreq, self).__init__()
        self.train_table = train_table

    def forward(self, indices):
        x = self.train_table[indices[:, 0], indices[:, 1]]
        return x

class LT(nn.Module):
    def __init__(self, train_table):
        super(LT, self).__init__()
        self.train_table = train_table
        self.train_table[self.train_table>0] = 1

    def forward(self, indices):
        x = self.train_table[indices[:, 0], indices[:, 1]]
        return x

class Random(nn.Module):
    def __init__(self, train_table):
        super(Random, self).__init__()
        
    def forward(self, indices):
        return torch.rand(indices.shape[0])

class Freq(nn.Module):
    def __init__(self, train_table):
        super(Freq, self).__init__()
        self.train_table = torch.sum(train_table, 1)

    def forward(self, indices):
        return self.train_table[indices[:, 1]]

class Mode(nn.Module):
    def __init__(self, train_table):
        super(Mode, self).__init__()
        self.train_table = torch.sum(train_table, 1)
        mode_ind = torch.argmax(self.train_table)
        self.train_table = torch.zeros(self.train_table.shape)
        self.train_table[mode_ind] = 1

    def forward(self, indices):
        return self.train_table[indices[:, 1]]