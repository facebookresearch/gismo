import copy
import os
import time 

import numpy as np
import torch
from data_loader import SubsData, load_data
from state_loader import create_output_dir, load_saved_models, save_model
from torch.utils.data import DataLoader
from models import *

class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()

    def get_loss(self, model, indices, nr, context, name):
        sims = self.get_model_output(model, indices, context, nr, name)
        sims = sims.view(-1, nr + 1)
        return sims

    def embed_context(self, embeddings, indices, nr):
        embeddings_indices = embeddings[0 : -1 : nr + 1]
        context_emb = torch.sum(embeddings_indices, dim=1)
        mask = indices[0 : -1 : nr + 1] > 0
        norm = torch.sum(mask, 1).view(-1, 1)
        context_emb = context_emb / norm
        return context_emb.repeat(1, nr + 1).view(
            embeddings.shape[0], embeddings.shape[2]
        )

    def embed_context_slow(self, embeddings, indices, nr):
        context_emb = torch.sum(embeddings, dim=1)
        mask = indices > 0
        norm = torch.sum(mask, 1).view(-1, 1)
        return context_emb / norm

    def get_model_output(self, model, indices, context, nr, name, embeddings=None):
        if name == "GCN" or name == "SAGE" or name == "GIN" or name == "GAT":
            if embeddings is None:
                embeddings = model()

            if context:
                embs = torch.index_select(
                    embeddings, dim=0, index=indices.reshape(-1)
                ).view(indices.shape[0], indices.shape[1], -1)
                context_emb = self.embed_context(embs[:, 2:], indices[:, 2:], nr)
                ing1 = embs[:, 0] + context_emb
                ing2 = embs[:, 1] + context_emb
            else:
                embs = embeddings[indices[:, :2]]
                ing1 = embs[:, 0]
                ing2 = embs[:, 1]
            # compute cosine similarities
            # sims = cos_layer(ing1, ing2)
            sims = torch.bmm(ing1.unsqueeze(1), ing2.unsqueeze(2))
        elif name == "GIN_MLP":
            sims = model(indices, context)
        elif name == "MLP":
            sims = model(indices, context)
        elif name == "MLP_CAT":
            if context:
                sims = model(indices)
            else:
                print("The model MLP_CAT is not defined in the context-free setup")
                exit()
        elif name == "MLP_ATT":
            if context:
                sims = model(indices, nr)
            else:
                print("The model MLP_ATT is not defined in the context-free setup")
                exit()
        return sims

    def get_rank(self, scores):
        mask = scores[:, 0].repeat(scores.shape[1]).view(scores.shape[1], -1).T
        predicted_index = torch.max(scores, dim=1).indices
        ranks = torch.sum(scores >= mask, 1) - 1
        ranks[ranks < 1] = 1
        return ranks, predicted_index

    def get_loss_test(self, model, dataloader, n_ingrs, context, name):
        # rank_file = open("/private/home/baharef/inversecooking2.0/proposed_model/GIN_MLP_foodbert_rankings.txt", "w")
        if name == "GCN":
            embeddings = model()
        else:
            embeddings = None

        mrr = 0.0
        hits = {1: 0, 3: 0, 10: 0}
        counter = 0

        for batch in dataloader:
            sims = self.get_model_output(
                model, batch, context, n_ingrs - 1, name, embeddings
            ).view(-1, n_ingrs)
            ranks, predicted_index = self.get_rank(sims)
            mrr += torch.sum(1.0 / ranks)
            for key in hits:
                hits[key] += torch.sum(ranks <= key)
            counter += len(ranks)

            # for ind in range(ranks.shape[0]):
            #     rank_file.write(str(batch[ind*n_ingrs][0].cpu().item()) + " " + str(batch[ind*n_ingrs][1].cpu().item()) + " " + str(ranks[ind].cpu().item()) + " " + str(batch[ind*n_ingrs+predicted_index[ind].cpu().item()][1].cpu().item()) + "\n")

        counter = float(counter)
        mrr = float(mrr) * 100
        mrr /= counter
        for key in hits:
            hits[key] = float(hits[key])
            hits[key] /= counter / 100.0
        return mrr, hits

    def train_classification_gcn(
        self, adj, train_dataloader, val_dataloader, test_dataloader, n_ingrs, cfg, node_count2id, node_id2name
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = globals()[cfg.name](
            in_channels=cfg.emb_d,
            hidden_channels=cfg.hidden,
            num_layers=cfg.nlayers,
            dropout=cfg.dropout,
            adj=adj,
            device=device,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.w_decay)
        # cos_layer = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        base_dir = os.path.join(
            "/checkpoint/baharef", cfg.setup, cfg.name, "sept-21/checkpoints/"
        )
        context = 1 if cfg.setup == "context-full" or cfg.setup == "context_full" else 0
        output_dir = create_output_dir(base_dir, cfg)

        best_val_mrr = 0
        best_model = None

        model, opt, best_model = load_saved_models(output_dir, model, opt)

        if best_model:
            best_val_mrr = best_model.mrr
            best_model = best_model.to(device)
            best_model.eval()

        model = model.to(device)
        model.epoch = model.epoch.to(device)

        loss_layer = torch.nn.CrossEntropyLoss()

        for epoch in range(model.epoch.cpu().item() + 1, cfg.epochs + 1):
            model.train()
            epoch_loss = 0.0
            start_time = time.time()
            for train_batch in train_dataloader:
                indices = train_batch[:, :-1]
                sims = self.get_loss(model, indices, cfg.nr, context, cfg.name)
                targets = torch.zeros(sims.shape[0]).long().to(device)
                loss = loss_layer(sims, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.cpu().item()
            print(time.time()-start_time)
            print(epoch, epoch_loss)
            # print("eps", model.layers[0].eps.cpu().item())
            model.epoch.data = torch.from_numpy(np.array([epoch])).to(device)
            save_model(model, opt, output_dir, is_best_model=False)
            if epoch % cfg.val_itr == 0:
                with torch.no_grad():
                    model.eval()
                    val_mrr, val_hits = self.get_loss_test(
                        model, val_dataloader, n_ingrs, context, cfg.name
                    )
                    print("vall metrics:", val_mrr, val_hits)
                    if val_mrr > best_val_mrr:
                        best_val_mrr = val_mrr
                        best_model = copy.deepcopy(model)
                        print("Best val mrr updated to", best_val_mrr)
                        best_model.mrr.data = torch.from_numpy(
                            np.array([best_val_mrr])
                        ).to(device)
                        best_model.epoch.data = torch.from_numpy(np.array([epoch])).to(
                            device
                        )
                        save_model(best_model, opt, output_dir, is_best_model=True)

        print("Training finished!")

        with torch.no_grad():
            best_model.eval()
            test_mrr, test_hits = self.get_loss_test(
                best_model, test_dataloader, n_ingrs, context, cfg.name
            )
        print(test_mrr, test_hits)
        return best_val_mrr, test_mrr, test_hits

    def train_recommender_gcn(self, cfg):

        graph, train_dataset, val_dataset, test_dataset, ingrs, node_count2id, node_id2name, node_id2count = load_data(
            cfg.nr,
            cfg.max_context,
            cfg.add_self_loop,
            dir_="/private/home/baharef/inversecooking2.0/data/flavorgraph",
        )
        n_ingrs = len(ingrs)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=SubsData.collate_fn,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=int(cfg.val_test_batch_size),
            shuffle=True,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=SubsData.collate_fn_val_test,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=int(cfg.val_test_batch_size),
            shuffle=True,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=SubsData.collate_fn_val_test,
        )

        val_mrr_arr = []
        test_mrr_arr = []
        test_hits_arr = []
        for trial in range(cfg.ntrials):
            val_mrr, test_mrr, test_hits = self.train_classification_gcn(
                graph, train_dataloader, val_dataloader, test_dataloader, n_ingrs, cfg, node_count2id, node_id2name
            )
            val_mrr_arr.append(val_mrr)
            test_mrr_arr.append(test_mrr)
            test_hits_arr.append(test_hits)

        print(val_mrr_arr, test_mrr_arr, test_hits)
