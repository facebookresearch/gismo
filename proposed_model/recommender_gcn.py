import os
import copy
import time
import numpy as np
import torch
from state_loader import create_output_dir, load_saved_models, save_model
from torch.utils.data import DataLoader

from data_loader import SubsData, load_data
from models import GCN, MLP


class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()

    def get_loss(self, embeddings, indices, nr, context, cos_layer):
        # cos_layer.train()
        sims = self.get_model_output(embeddings, indices, context, nr, cos_layer)
        sims = sims.view(-1, nr + 1)
        return sims

    def embed_context(self, embeddings, indices, nr):
        embeddings_indices = embeddings[0:-1:nr+1]
        context_emb = torch.sum(embeddings_indices, dim=1)
        mask = indices[0:-1:nr+1] > 0
        norm = torch.sum(mask, 1).view(-1, 1)
        context_emb = context_emb/norm
        return context_emb.repeat(1, nr+1).view(embeddings.shape[0], embeddings.shape[2])
    
    def embed_context_slow(self, embeddings, indices, nr):
        context_emb = torch.sum(embeddings, dim=1)
        mask = indices > 0
        norm = torch.sum(mask, 1).view(-1, 1)
        return context_emb/norm


    def get_model_output(self, embeddings, indices, context, nr, cos_layer):
        if context:
            embs = embeddings[indices]
            context_emb = self.embed_context(embs[:, 2:], indices[:, 2:], nr)
            ing1 = embs[:, 0] + context_emb
            ing2 = embs[:, 1] + context_emb
        else:
            embs = embeddings[indices[:,:2]]
            ing1 = embs[:, 0]
            ing2 = embs[:, 1]

        # compute cosine similarities
        # sims = cos_layer(ing1, ing2)

        # compute dot-product similarities
        sims = torch.bmm(ing1.unsqueeze(1), ing2.unsqueeze(2))

        return sims

    def get_rank(self, scores):
        mask = scores[:, 0].repeat(scores.shape[1]).view(scores.shape[1], -1).T
        ranks = torch.sum(scores >= mask, 1) - 1
        ranks[ranks < 1] = 1
        return ranks

    def get_loss_test(self, embeddings, dataloader, n_ingrs, context, cos_layer):
        cos_layer.eval()
        mrr = 0.0
        hits = {1: 0, 3: 0, 10: 0}
        counter = 0

        for batch in dataloader:
            sims = self.get_model_output(embeddings, batch, context, n_ingrs-1, cos_layer).view(
                batch.shape[0] // n_ingrs, -1
            )
            ranks = self.get_rank(sims)
            mrr += torch.sum(1.0 / ranks)
            for key in hits:
                hits[key] += torch.sum(ranks <= key)
            counter += len(ranks)
        counter = float(counter)
        mrr = float(mrr) * 100
        mrr /= counter
        for key in hits:
            hits[key] = float(hits[key])
            hits[key] /= counter / 100.0
        return mrr, hits

    def train_classification_gcn(
        self, adj, train_dataloader, val_dataloader, test_dataloader, n_ingrs, cfg
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

        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        cos_layer = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        base_dir = os.path.join("/checkpoint/baharef", cfg.setup , cfg.name, "aug-30/checkpoints/")
        context = 1 if cfg.setup == "context-full" else 0
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
            start_time = time.time()
            model.train()
            epoch_loss = 0.0
            for train_batch in train_dataloader:
                embeddings = model()
                indices = train_batch[:, :-1]
                sims = self.get_loss(embeddings, indices, cfg.nr, context, cos_layer)
                targets = torch.zeros(sims.shape[0]).long().to(device)
                loss = loss_layer(sims, targets) + cfg.w_decay * torch.norm(embeddings)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.cpu().item()
            print(epoch, epoch_loss)
            print(time.time()-start_time)
            model.epoch.data = torch.from_numpy(np.array([epoch])).to(device)
            save_model(model, opt, output_dir, is_best_model=False)
            if epoch % cfg.val_itr == 0:
                with torch.no_grad():
                    model.eval()
                    embeddings = model()
                    val_mrr, val_hits = self.get_loss_test(
                        embeddings, val_dataloader, n_ingrs, context, cos_layer
                    )
                    print("vall metrics:", val_mrr, val_hits)
                    if val_mrr > best_val_mrr:
                        best_val_mrr = val_mrr
                        best_model = copy.deepcopy(model)
                        print("Best val mrr updated to", best_val_mrr)
                        best_model.mrr.data = torch.from_numpy(np.array([best_val_mrr])).to(
                            device
                        )
                        best_model.epoch.data = torch.from_numpy(np.array([epoch])).to(
                            device
                        )
                        save_model(best_model, opt, output_dir, is_best_model=True)

        print("Training finished!")

        with torch.no_grad():
            best_model.eval()
            best_embeddings = best_model()
            test_mrr, test_hits = self.get_loss_test(
                best_embeddings, test_dataloader, n_ingrs, context, cos_layer
            )
        print(test_mrr, test_hits)
        return best_val_mrr, test_mrr, test_hits

    def train_recommender_gcn(self, cfg):
        graph, train_dataset, val_dataset, test_dataset, ingrs, _, _, _ = load_data(
            cfg.nr, cfg.max_context, dir_="/private/home/baharef/inversecooking2.0/data/flavorgraph"
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
            batch_size=cfg.val_test_batch_size,
            shuffle=True,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=SubsData.collate_fn_val_test,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=cfg.val_test_batch_size,
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
                graph, train_dataloader, val_dataloader, test_dataloader, n_ingrs, cfg
            )
            val_mrr_arr.append(val_mrr)
            test_mrr_arr.append(test_mrr)
            test_hits_arr.append(test_hits)

        print(val_mrr_arr, test_mrr_arr, test_hits)
