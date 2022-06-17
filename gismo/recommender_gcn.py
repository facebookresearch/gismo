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

    def get_loss(self, model, indices, nr, context, name, margin):
        sims = self.get_model_output(model, indices, context, nr, name)
        sims = sims.view(-1, nr + 1)

        if name == "SAGE" or name == "GIN":
            sims = torch.relu(sims[:, 1] - sims[:, 0] + margin)
            loss = torch.sum(sims)
            return loss

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
            sims = torch.bmm(ing1.unsqueeze(1), ing2.unsqueeze(2))
        elif name == "GIN_MLP" or name == "GIST":
            sims = model(indices, context, nr)
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

        return ranks, predicted_index

    def get_rank_filter(self, scores, filter_mask):
        mask = scores[:, 0].repeat(scores.shape[1]).view(scores.shape[1], -1).T
        scores = scores + filter_mask
        predicted_index = torch.max(scores, dim=1).indices
        ranks = torch.sum(scores >= mask, 1) 
        return ranks, predicted_index

    def get_loss_naive_baseline(self, dataloader, model, n_ingrs, rank_file_path, filter):
        rank_file = open(rank_file_path, "w")
        pred_file_path = os.path.splitext(rank_file_path)[0] + "_full.txt"
        pred_file = open(pred_file_path, "w")
        print(pred_file_path)

        mrr = 0.0
        hits = {1: 0, 3: 0, 10: 0}
        counter = 0
        
        for tup in dataloader:
            batch = tup[0]
            filtered_mask = tup[1]
            sims = model(batch).view(-1, n_ingrs+1)
            
            if filter:
                ranks, predicted_index = self.get_rank_filter(sims, filtered_mask)
            else:
                ranks, predicted_index = self.get_rank(sims)
            
            # TODO_QUENTIN: hack to get the top predictions and not just the first
            top_k = 10
            top_k_indices = sims.topk(top_k, dim=-1, largest=True, sorted=True).indices
            print(top_k_indices)

            mrr += torch.sum(1.0 / ranks)
            for key in hits:
                hits[key] += torch.sum(ranks <= key)
            counter += len(ranks)
            
            for ind in range(ranks.shape[0]):
                ingr_a = batch[ind*(n_ingrs+1)][0].cpu().item()
                ingr_b = batch[ind*(n_ingrs+1)][1].cpu().item()
                rank_of_ingr_b = ranks[ind].cpu().item()
                pred_ingr = batch[ind*(n_ingrs+1)+predicted_index[ind].cpu().item()][1].cpu().item()
                line = [str(ingr_a), str(ingr_b), str(rank_of_ingr_b), str(pred_ingr)]
                rank_file.write(" ".join(line) + "\n")

                line = [str(ingr_a), str(pred_ingr)]
                for k in range(top_k):
                    pred_ingr = batch[ind*(n_ingrs+1)+top_k_indices[ind][k].cpu().item()][1].cpu().item()
                    line.append(str(pred_ingr))
                pred_file.write(" ".join(line) + "\n")

            '''
            for ind in range(ranks.shape[0]):
                rank_file.write(str(batch[ind*n_ingrs][0].cpu().item()) + " " + str(batch[ind*n_ingrs][1].cpu().item()) + " " + str(ranks[ind].cpu().item()) + " " + str(batch[ind*n_ingrs+predicted_index[ind].cpu().item()][1].cpu().item()) + "\n")
            '''

        counter = float(counter)
        mrr = float(mrr) * 100
        mrr /= counter
        for key in hits:
            hits[key] = float(hits[key])
            hits[key] /= counter / 100.0
        return mrr, hits

    def get_loss_test(self, model, dataloader, n_ingrs, context, name, rank_file_path, filter):
        rank_file = open(rank_file_path, "w")
        pred_file_path = os.path.splitext(rank_file_path)[0] + "_full.txt"
        pred_file = open(pred_file_path, "w")
        print(pred_file_path)

        if name == "GCN" or name == "SAGE" or name == "GIN":
            embeddings = model()
        else:
            embeddings = None

        mrr = 0.0
        hits = {1: 0, 3: 0, 10: 0}
        counter = 0

        for tup in dataloader:
            batch = tup[0]
            filter_mask = tup[1]
            sims = self.get_model_output(
                model, batch, context, n_ingrs, name, embeddings
            ).view(-1, n_ingrs + 1)
            
            if not filter:
                ranks, predicted_index = self.get_rank(sims)
            else:
                ranks, predicted_index = self.get_rank_filter(sims, filter_mask)
            
            # TODO_QUENTIN: hack to have the top 5 suggestions from GISMO
            top_k = 10
            top_k_indices = sims.topk(top_k, dim=-1, largest=True, sorted=True).indices

            mrr += torch.sum(1.0 / ranks)
            for key in hits:
                hits[key] += torch.sum(ranks <= key)
            counter += len(ranks)

            for ind in range(ranks.shape[0]):
                ingr_a = batch[ind*(n_ingrs+1)][0].cpu().item()
                ingr_b = batch[ind*(n_ingrs+1)][1].cpu().item()
                rank_of_ingr_b = ranks[ind].cpu().item()
                pred_ingr = batch[ind*(n_ingrs+1)+predicted_index[ind].cpu().item()][1].cpu().item()
                line = [str(ingr_a), str(ingr_b), str(rank_of_ingr_b), str(pred_ingr)]
                rank_file.write(" ".join(line) + "\n")

                line = [str(ingr_a), str(pred_ingr)]
                for k in range(top_k):
                    pred_ingr = batch[ind*(n_ingrs+1)+top_k_indices[ind][k].cpu().item()][1].cpu().item()
                    line.append(str(pred_ingr))
                pred_file.write(" ".join(line) + "\n")

        counter = float(counter)
        mrr = float(mrr) * 100
        mrr /= counter
        for key in hits:
            hits[key] = float(hits[key])
            hits[key] /= counter / 100.0
        return mrr, hits

    def train_classification_gcn(
        self, adj, train_dataloader, val_dataloader, test_dataloader, n_ingrs, cfg, node_count2id, node_id2name, recipe_id2counter, device, ingrs, lookup_table
    ):
        base_dir = os.path.join(
            "/checkpoint/baharef", cfg.setup, cfg.name, "oct-26/checkpoints/"
        )
        context = 1 if cfg.setup == "context-full" or cfg.setup == "context_full" else 0
        # output_dir = create_output_dir(base_dir, cfg)
        output_dir = create_output_dir(
                os.path.join("/private/home/qduval/baharef/out"),
                cfg,
        )
        ranks_file_val = os.path.join(output_dir, "val_ranks.txt")
        ranks_file_test = os.path.join(output_dir, "test_ranks.txt")

        if cfg.name == "LT" or cfg.name == "LTFreq" or cfg.name == "Random" or cfg.name == "Freq" or cfg.name == "Mode":
            model = globals()[cfg.name](lookup_table)
            model = model.to(device)
            print("No training is involved for this setup!")
            # val_mrr, val_hits = self.get_loss_naive_baseline(
            #     val_dataloader, model, n_ingrs, ranks_file_val, cfg.filter
            # )
            test_mrr, test_hits = self.get_loss_naive_baseline(
                test_dataloader, model, n_ingrs, ranks_file_test, cfg.filter
            )
            print(test_mrr, test_hits)

            return 0.0, test_mrr, test_hits

        model = globals()[cfg.name](
            in_channels=cfg.emb_d,
            hidden_channels=cfg.hidden,
            num_layers=cfg.nlayers,
            dropout=cfg.dropout,
            adj=adj,
            device=device,
            recipe_id2counter=recipe_id2counter,
            with_titles=cfg.with_titles,
            with_set=cfg.with_set,
            node_count2id=node_count2id,
            init_emb=cfg.init_emb,
            context_emb_mode=cfg.context_emb_mode,
            pool=cfg.pool
        ).to(device)


        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.w_decay)
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
                sims = self.get_loss(model, indices, cfg.nr, context, cfg.name, cfg.margin)
                
                if cfg.name != "SAGE" and cfg.name != "GIN":
                    targets = torch.zeros(sims.shape[0]).long().to(device)
                    loss = loss_layer(sims, targets)
                else:
                    loss = sims

                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if cfg.init_emb == "flavorgraph2" or cfg.init_emb == "food_bert2":
                    loss += torch.norm(model.ndata1.weight) * cfg.lambda_

                epoch_loss += loss.cpu().item()

            print(epoch, epoch_loss)
            print(time.time() - start_time)
            model.epoch.data = torch.from_numpy(np.array([epoch])).to(device)
            
            save_model(model, opt, output_dir, is_best_model=False)
            
            if epoch % cfg.val_itr == 0:
                with torch.no_grad():
                    model.eval()
                    val_mrr, val_hits = self.get_loss_test(
                        model, val_dataloader, n_ingrs, context, cfg.name, ranks_file_val, cfg.filter
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
        "Val results:"
        with torch.no_grad():
            best_model.eval()
            val_mrr, val_hits = self.get_loss_test(
                best_model, val_dataloader, n_ingrs, context, cfg.name, ranks_file_val, cfg.filter
            )
        print(val_mrr, val_hits)
        "Test results:"
        with torch.no_grad():
            best_model.eval()
            test_mrr, test_hits = self.get_loss_test(
                best_model, test_dataloader, n_ingrs, context, cfg.name, ranks_file_test, cfg.filter
            )

        print(test_mrr, test_hits)
        return best_val_mrr, test_mrr, test_hits

    def train_recommender_gcn(self, cfg):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        graph, train_dataset, val_dataset, test_dataset, ingrs, node_count2id, node_id2name, node_id2count, recipe_id2counter, filter_ingredient = load_data(
            cfg.nr,
            cfg.max_context,
            cfg.add_self_loop,
            cfg.neg_sampling,
            cfg.data_augmentation,
            cfg.p_augmentation,
            cfg.filter,
            device,
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
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=SubsData.collate_fn_val_test,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=int(cfg.val_test_batch_size),
            shuffle=False,
            sampler=None,
            batch_sampler=None,
            num_workers=0,
            collate_fn=SubsData.collate_fn_val_test,
        )

        for trial in range(cfg.ntrials):
            val_mrr, test_mrr, test_hits = self.train_classification_gcn(
                graph, train_dataloader, val_dataloader, test_dataloader, n_ingrs, cfg, node_count2id, node_id2name, recipe_id2counter, device, ingrs, train_dataset.lookup_table
            )
            print(val_mrr, test_mrr, test_hits)
