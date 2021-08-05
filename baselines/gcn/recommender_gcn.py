import copy
import numpy as np
import torch
import torch.nn.functional as F

from baselines.gcn.data_loader import SubsData, load_data
from baselines.gcn.models import GCN
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self):
        super(Trainer, self).__init__()

    def get_loss(self, embeddings, indices, labels):
        dist = self.get_model_output(embeddings, indices)
        loss = F.mse_loss(dist, labels.float(), reduction='mean')
        return loss

    def get_model_output(self, embeddings, indices):
        # embeddings = model()
        x = torch.index_select(embeddings, 0, indices[:, 0])
        y = torch.index_select(embeddings, 0, indices[:, 1])
        # dist = torch.sum((x - y) ** 2, 1)/embeddings.shape[1]
        # dist = torch.sigmoid(dist)

        # dist = torch.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        dist = torch.norm(x - y, 2, dim=1)
        return dist

    def get_rank(self, scores):
        mask = scores[:, 0].repeat(scores.shape[1]).view(scores.shape[1], -1).T
        ranks = torch.sum(scores < mask, 1) + 1
        return ranks

    def get_loss_test(self, embeddings, dataloader, n_ingrs):
        mrr = 0.0
        hits = {1:0, 3:0, 10:0}
        counter = 0
        # model.eval()
        for batch in dataloader:
            if counter % 1000 == 0:
                print(counter)
            scores = self.get_model_output(embeddings, batch).view(batch.shape[0]//n_ingrs, -1)
            ranks = self.get_rank(scores)
            mrr += torch.sum(1.0/ranks)
            for key in hits:
                hits[key] += torch.sum(ranks <= key)
            counter += len(ranks)
        counter = float(counter)
        mrr = float(mrr)
        mrr /= counter
        for key in hits:
            hits[key] = float(hits[key])
            hits[key] /= (counter/100.0)
        return mrr, hits
                

    def train_classification_gcn(self, adj, train_dataloader, val_dataloader, test_dataloader, n_ingrs, cfg):
        model = GCN(in_channels=cfg.emb_d, hidden_channels=cfg.hidden, num_layers=cfg.nlayers, dropout=cfg.dropout, adj=adj)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.w_decay)
        best_val_mrr = 0
        best_model = None
        if torch.cuda.is_available():
            model = model.cuda()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for train_batch in train_dataloader:
                embeddings = model()
                indices = train_batch[:,:-1]
                labels = train_batch[:, -1]
                loss = self.get_loss(embeddings, indices, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.cpu().item()
            print(epoch, epoch_loss)
            if epoch % cfg.val_itr == 0:
                model.eval()
                embeddings = model()
                val_mrr, val_hits = self.get_loss_test(embeddings, val_dataloader, n_ingrs)
                print(val_mrr, val_hits)
                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    best_model = copy.deepcopy(model)
                    print("Best val mrr updated to", best_val_mrr)

        best_model.eval()
        best_embeddings = best_model()
        test_mrr, test_hits = self.get_loss_test(best_embeddings, test_dataloader, n_ingrs)

        return val_mrr, test_mrr, test_hits

    def train_knn_gcn(self, cfg):
        graph, train_dataset, val_dataset, test_dataset, n_ingrs = load_data(cfg.nr, dir_ = '/private/home/baharef/inversecooking2.0/data/flavorgraph')
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, sampler=None,
                                        batch_sampler=None, num_workers=0, collate_fn=SubsData.collate_fn)

        val_dataloader = DataLoader(val_dataset, batch_size=cfg.val_test_batch_size, shuffle=True, sampler=None,
                                        batch_sampler=None, num_workers=0, collate_fn=SubsData.collate_fn_val_test)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.val_test_batch_size, shuffle=True, sampler=None,
                                        batch_sampler=None, num_workers=0, collate_fn=SubsData.collate_fn_val_test)
                  
        val_mrr_arr = []
        test_mrr_arr = []
        test_hits_arr = []
        for trial in range(cfg.ntrials):
            val_mrr, test_mrr, test_hits = self.train_classification_gcn(graph, train_dataloader, val_dataloader, test_dataloader, n_ingrs, cfg)
            val_mrr_arr.append(val_mrr)
            test_mrr_arr.append(test_mrr)
            test_hits_arr.append(test_hits)

        print(val_mrr_arr, test_mrr_arr, test_hits)
