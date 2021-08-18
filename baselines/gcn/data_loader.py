import os
import csv
import copy
import numpy as np
import dgl
import torch
import json
import pickle
import random
import dgl.ops as ops
import dgl.function as fn
import torch.utils.data as data
from inv_cooking.datasets.vocabulary import Vocabulary
from typing import Tuple

def load_edges(dir_, node_id2count, node_count2id, node_id2name, nnodes, normalize=True):
    sources, destinations, weights, types = [], [], [], []

    with open(os.path.join(dir_, 'edges_191120.csv'), 'r') as edges_file:
        csv_reader = csv.DictReader(edges_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            node1, node2 = int(row["id_1"]), int(row["id_2"])
            node1_cnt, node2_cnt = node_id2count[node1], node_id2count[node2]

            edge_type = row["edge_type"]
            if 'ingr-ingr' in row["edge_type"]:
                edge_type = 1
                score = float(row["score"])
            elif 'ingr-fcomp' in row["edge_type"]:
                edge_type = 2
                score = 1
            elif 'ingr-dcomp' in row["edge_type"]:
                edge_type = 3
                score = 1
            sources.append(node1_cnt)
            destinations.append(node2_cnt)
            weights.append(score)
            types.append(edge_type)

            # make it symmetric
            sources.append(node2_cnt)
            destinations.append(node1_cnt)
            weights.append(score)
            types.append(edge_type)

    # add self-loop        
    for node in range(nnodes):
        sources.append(node)
        destinations.append(node)
        weights.append(1)
        types.append(4)

    sources = torch.tensor(sources)
    destinations = torch.tensor(destinations)
    weights = torch.tensor(weights)
    types = torch.tensor(types)

    if torch.cuda.is_available():
        sources = sources.cuda()
        destinations = destinations.cuda()
        weights = weights.cuda()
        types = types.cuda()

    graph = dgl.graph((sources, destinations))
    graph.edata['w'] = weights
    graph.edata['t'] = types

    # symmetric normalization
    if normalize:
        in_degree = ops.copy_e_sum(graph, graph.edata['w'])
        in_norm = torch.pow(in_degree, -0.5)
        out_norm = torch.pow(in_degree, -0.5).unsqueeze(-1)
        graph.ndata['in_norm'] = in_norm
        graph.ndata['out_norm'] = out_norm
        graph.apply_edges(fn.u_mul_v('in_norm', 'out_norm', 'n'))
        graph.edata['w'] = graph.edata['w'] * graph.edata['n'].squeeze()

    return graph

def load_nodes(dir_):
    node_id2name = {}
    node_name2id = {}
    node_id2type = {}
    ingredients_cnt = []
    compounds_cnt = []
    node_id2count = {}
    node_count2id = {}
    counter = 0
    with open(os.path.join(dir_, 'nodes_191120.csv'), 'r') as nodes_file:
        csv_reader = csv.DictReader(nodes_file)
        for row in csv_reader:
            node_id = int(row["node_id"])
            node_type = row["node_type"]
            node_id2name[node_id] = row["name"]
            node_name2id[row["name"]] = node_id
            node_id2type[node_id] = node_type
            if 'ingredient' in node_type:
                ingredients_cnt.append(counter)
            else:
                compounds_cnt.append(counter)
            node_id2count[node_id] = counter
            node_count2id[counter] = node_id
            counter += 1
    nnodes = len(node_id2name)
    print("#nodes:", nnodes)
    print("#ingredient nodes:", len(ingredients_cnt))
    print("#compound nodes:", len(compounds_cnt))
    return node_id2count, node_count2id, node_id2name, node_name2id, ingredients_cnt, node_id2name, nnodes

def node_count2name(count, node_count2id, node_id2name):
    return node_id2name[node_count2id[count]]

def load_graph(dir_):
    node_id2count, node_count2id, node_id2name, node_name2id, ingredients_cnt, node_id2name, nnodes = load_nodes(dir_)
    graph = load_edges(dir_, node_id2count, node_count2id, node_id2name, nnodes)
    return graph, node_name2id, node_id2count, ingredients_cnt, node_count2id, node_id2name

 
def load_data(nr, dir_):
    graph, node_name2id, node_id2count, ingredients_cnt, node_count2id, node_id2name = load_graph(dir_)
    train_dataset = SubsData('/private/home/baharef/inversecooking2.0/new/old/', 'train', node_name2id, node_id2count, ingredients_cnt, nr)
    val_dataset = SubsData('/private/home/baharef/inversecooking2.0/new/old/', 'val', node_name2id, node_id2count, ingredients_cnt, nr)
    test_dataset = SubsData('/private/home/baharef/inversecooking2.0/new/old', 'test', node_name2id, node_id2count, ingredients_cnt, nr)
    
    return graph, train_dataset, val_dataset, test_dataset, len(ingredients_cnt), node_count2id, node_id2name, node_id2count


class SubsData(data.Dataset):
    def __init__(self, data_dir: str, split: str, node_name2id: dict, node_id2count: dict, ingredients_cnt: list, nr: int, pre_processed_dir='/private/home/baharef/inversecooking2.0/new'):
        self.substitutions_dir = os.path.join(data_dir, split + '_comments_subs.txt')
        self.split = split
        self.ingr_vocab = Vocabulary()
        self.dataset = []
        self.nr = nr
        self.pre_processed_dir = pre_processed_dir
        # load ingredient voc
        self.ingr_vocab = pickle.load(open(os.path.join(self.pre_processed_dir, "final_recipe1m_vocab_ingrs.pkl"),"rb",))
        self.node_name2id = node_name2id
        self.node_id2count = node_id2count
        self.ingredients_cnt = ingredients_cnt
        # load dataset

        self.dataset_list = json.load(open(self.substitutions_dir, 'r'))
        self.dataset = self.context_free_examples(self.dataset_list, self.ingr_vocab)

        print("Number of datapoints in", self.split, self.dataset.shape[0])

    def context_free_examples(self, examples, vocabs):
        output = torch.zeros(len(examples), 2)
        for ind, example in enumerate(examples):
            subs = example['subs']
            r_name1 = vocabs.idx2word[vocabs.word2idx[subs[0]]][0]
            r_name2 = vocabs.idx2word[vocabs.word2idx[subs[1]]][0]

            subs = torch.tensor([self.node_id2count[self.node_name2id[r_name1]], self.node_id2count[self.node_name2id[r_name2]]])
            output[ind, :] = subs
        return output

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.split == 'train':
            pos_example = self.dataset[index, :].view(1, 2)
            neg_examples = self.neg_examples(pos_example, self.nr, self.ingredients_cnt)
            pos_labels = torch.zeros(1, 1)
            neg_labels = torch.ones(len(neg_examples), 1)
            pos_batch = torch.cat((pos_example, pos_labels), 1)
            neg_batch = torch.cat((neg_examples, neg_labels), 1)
            ret = torch.cat((pos_batch, neg_batch), 0)

            return ret.long()

        elif self.split == 'val' or self.split == 'test':
            pos_example = self.dataset[index, :]
            all_indices = self.ingredients_cnt.copy()
            all_indices.remove(pos_example[1])
            neg_examples_1 = torch.tensor(all_indices).view(-1, 1)
            neg_examples_0 = self.dataset[index, :][0].repeat(len(all_indices)).view(-1, 1)
            neg_examples = torch.cat((neg_examples_0, neg_examples_1), 1)
            return torch.cat((pos_example.view(1, 2), neg_examples), 0)

    @staticmethod
    def collate_fn(data):
        n_examples = data[0].shape[0]
        batch = torch.zeros(len(data)*data[0].shape[0], data[0].shape[1])
        for ind, sub_batch in enumerate(data):
            batch[ind*n_examples:(ind+1)*n_examples, :] = sub_batch
        if torch.cuda.is_available():
            batch = batch.cuda()
        return batch.long()

    @staticmethod
    def collate_fn_val_test(data):
        batch = torch.zeros(len(data)*data[0].shape[0], data[0].shape[1])
        for ind, sub_batch in enumerate(data):
            batch[ind*data[0].shape[0]:(ind+1)*data[0].shape[0], :] = sub_batch
        if torch.cuda.is_available():
            batch = batch.cuda()
        return batch.long()

    def neg_examples(self, example, nr, ingredients_cnt):
        neg_batch = torch.zeros(nr, 2)
        random_entities = torch.tensor(random.sample(ingredients_cnt, nr))
        neg_batch = torch.cat((example[0, 0].repeat(nr).view(nr, 1), random_entities.view(nr, 1)),1)
        return neg_batch

if __name__ == "__main__":
    graph, train_dataset, val_dataset, test_dataset, x, node_count2id, node_id2name, node_id2count = load_data(2,'/private/home/baharef/inversecooking2.0/data/flavorgraph')
