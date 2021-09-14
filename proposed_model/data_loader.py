import csv
import json
import os
import pickle
import random

import dgl
import dgl.function as fn
import dgl.ops as ops
import torch
import torch.utils.data as data

from inv_cooking.datasets.vocabulary import Vocabulary


def load_edges(
    dir_, node_id2count, node_count2id, node_id2name, nnodes, add_self_loop, normalize=True
):
    sources, destinations, weights, types = [], [], [], []

    with open(os.path.join(dir_, "edges_191120.csv"), "r") as edges_file:
        csv_reader = csv.DictReader(edges_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(f'Column names are {", ".join(row)}')
                line_count += 1
            node1, node2 = int(row["id_1"]), int(row["id_2"])
            node1_cnt, node2_cnt = node_id2count[node1], node_id2count[node2]

            edge_type = row["edge_type"]
            if "ingr-ingr" in row["edge_type"]:
                edge_type = 1
                score = float(row["score"])
            elif "ingr-fcomp" in row["edge_type"]:
                edge_type = 2
                score = 1
            elif "ingr-dcomp" in row["edge_type"]:
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
    if add_self_loop:
        for node in range(nnodes):
            sources.append(node+1)
            destinations.append(node+1)
            weights.append(1)
            types.append(4)
        print("self-loop is added to all nodes.")
        
    sources = torch.tensor(sources)
    destinations = torch.tensor(destinations)
    weights = torch.tensor(weights)
    types = torch.tensor(types)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        sources = sources.to(device)
        destinations = destinations.to(device)
        weights = weights.to(device)
        types = types.to(device)

    graph = dgl.graph((sources, destinations))
    graph.edata["w"] = weights
    graph.edata["t"] = types

    # symmetric normalization
    if normalize:
        in_degree = ops.copy_e_sum(graph, graph.edata["w"])
        in_norm = torch.pow(in_degree, -0.5)
        out_norm = torch.pow(in_degree, -0.5).unsqueeze(-1)
        graph.ndata["in_norm"] = in_norm
        graph.ndata["out_norm"] = out_norm
        graph.apply_edges(fn.u_mul_v("in_norm", "out_norm", "n"))
        graph.edata["w"] = graph.edata["w"] * graph.edata["n"].squeeze()

    return graph


def load_nodes(dir_):
    node_id2name = {}
    node_name2id = {}
    node_id2type = {}
    ingredients_cnt = []
    compounds_cnt = []
    node_id2count = {}
    node_count2id = {}
    counter = 1  # start with 1 to reserve 0 for padding
    with open(os.path.join(dir_, "nodes_191120.csv"), "r") as nodes_file:
        csv_reader = csv.DictReader(nodes_file)
        for row in csv_reader:
            node_id = int(row["node_id"])
            node_type = row["node_type"]
            node_id2name[node_id] = row["name"]
            node_name2id[row["name"]] = node_id
            node_id2type[node_id] = node_type
            if "ingredient" in node_type:
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
    return (
        node_id2count,
        node_count2id,
        node_id2name,
        node_name2id,
        ingredients_cnt,
        node_id2name,
        nnodes,
    )


def node_count2name(count, node_count2id, node_id2name):
    return node_id2name[node_count2id[count]]


def load_graph(add_self_loop, dir_):
    (
        node_id2count,
        node_count2id,
        node_id2name,
        node_name2id,
        ingredients_cnt,
        node_id2name,
        nnodes,
    ) = load_nodes(dir_)
    graph = load_edges(dir_, node_id2count, node_count2id, node_id2name, nnodes, add_self_loop)
    return (
        graph,
        node_name2id,
        node_id2count,
        ingredients_cnt,
        node_count2id,
        node_id2name,
    )


def load_data(nr, max_context, add_self_loop, dir_):
    (
        graph,
        node_name2id,
        node_id2count,
        ingredients_cnt,
        node_count2id,
        node_id2name,
    ) = load_graph(add_self_loop, dir_)
    ingr_vocabs = pickle.load(
        open(
            "/private/home/baharef/inversecooking2.0/data/substitutions/vocab_ingrs.pkl",
            "rb",
        )
    )
    train_dataset = SubsData(
        "/private/home/baharef/inversecooking2.0/data/substitutions/",
        "train",
        node_name2id,
        node_id2count,
        ingredients_cnt,
        nr,
        ingr_vocabs,
        max_context,
    )
    val_dataset = SubsData(
        "/private/home/baharef/inversecooking2.0/data/substitutions/",
        "val",
        node_name2id,
        node_id2count,
        ingredients_cnt,
        nr,
        ingr_vocabs,
        max_context,
    )
    test_dataset = SubsData(
        "/private/home/baharef/inversecooking2.0/data/substitutions/",
        "test",
        node_name2id,
        node_id2count,
        ingredients_cnt,
        nr,
        ingr_vocabs,
        max_context,
    )

    return (
        graph,
        train_dataset,
        val_dataset,
        test_dataset,
        ingredients_cnt,
        node_count2id,
        node_id2name,
        node_id2count,
    )


class SubsData(data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        node_name2id: dict,
        node_id2count: dict,
        ingredients_cnt: list,
        nr: int,
        vocab: Vocabulary,
        max_context: int,
    ):
        self.substitutions_dir = os.path.join(data_dir, split + "_comments_subs.txt")
        self.split = split
        self.dataset = []
        self.nr = nr
        self.max_context = max_context
        # load ingredient voc
        self.ingr_vocab = vocab
        self.node_name2id = node_name2id
        self.node_id2count = node_id2count
        self.ingredients_cnt = ingredients_cnt
        # load dataset
        self.dataset_list = json.load(open(self.substitutions_dir, "r"))
        self.dataset = self.context_full_examples(
            self.dataset_list, self.ingr_vocab, self.max_context
        )
        print("Number of datapoints in", self.split, self.dataset.shape[0])

    def context_full_examples(self, examples, vocabs, max_context):

        output = torch.full((len(examples), max_context + 2), 0)
        for ind, example in enumerate(examples):
            subs = example["subs"]
            context = example["ingredients"][:max_context]
            example["text"]
            r_name1 = vocabs.idx2word[vocabs.word2idx[subs[0]]][0]
            r_name2 = vocabs.idx2word[vocabs.word2idx[subs[1]]][0]

            context_ids = torch.empty(len(context))
            for ind_, ing in enumerate(context):
                context_ids[ind_] = self.node_id2count[
                    self.node_name2id[vocabs.idx2word[vocabs.word2idx[ing[0]]][0]]
                ]

            subs = torch.tensor(
                [
                    self.node_id2count[self.node_name2id[r_name1]],
                    self.node_id2count[self.node_name2id[r_name2]],
                ]
            )
            output[ind, :2] = subs
            output[ind, 2 : len(context) + 2] = context_ids

        return output

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        if self.split == "train":
            pos_example = self.dataset[index, :].view(1, -1)
            neg_examples = self.neg_examples(pos_example, self.nr, self.ingredients_cnt)
            pos_labels = torch.zeros(1, 1)
            neg_labels = torch.ones(len(neg_examples), 1)
            pos_batch = torch.cat((pos_example, pos_labels), 1)
            neg_batch = torch.cat((neg_examples, neg_labels), 1)
            ret = torch.cat((pos_batch, neg_batch), 0)

            return ret.long()

        elif self.split == "val" or self.split == "test":
            pos_example = self.dataset[index, :]
            all_indices = self.ingredients_cnt.copy()
            all_indices.remove(pos_example[1])
            neg_examples_1 = torch.tensor(all_indices).view(-1, 1)
            neg_examples_0 = (
                self.dataset[index, :][0].repeat(len(all_indices)).view(-1, 1)
            )
            neg_examples = torch.cat(
                (
                    neg_examples_0,
                    neg_examples_1,
                    pos_example[2:].repeat(len(all_indices), 1),
                ),
                1,
            )
            return torch.cat((pos_example.view(1, -1), neg_examples), 0)

    @staticmethod
    def collate_fn(data):
        n_examples = data[0].shape[0]
        batch = torch.zeros(len(data) * data[0].shape[0], data[0].shape[1])
        for ind, sub_batch in enumerate(data):
            batch[ind * n_examples : (ind + 1) * n_examples, :] = sub_batch
        if torch.cuda.is_available():
            batch = batch.cuda()
        return batch.long()

    @staticmethod
    def collate_fn_val_test(data):
        batch = torch.zeros(len(data) * data[0].shape[0], data[0].shape[1])
        for ind, sub_batch in enumerate(data):
            batch[ind * data[0].shape[0] : (ind + 1) * data[0].shape[0], :] = sub_batch
        if torch.cuda.is_available():
            batch = batch.cuda()
        return batch.long()

    def neg_examples(self, example, nr, ingredients_cnt):
        neg_batch = torch.zeros(nr, 2)
        random_entities = torch.tensor(random.sample(ingredients_cnt, nr))
        neg_batch = torch.cat(
            (example[0, 0].repeat(nr).view(nr, 1), random_entities.view(nr, 1)), 1
        )
        output = torch.cat((neg_batch, example[0, 2:].repeat(nr, 1)), 1)
        return output


if __name__ == "__main__":
    (
        graph,
        train_dataset,
        val_dataset,
        test_dataset,
        x,
        node_count2id,
        node_id2name,
        node_id2count,
    ) = load_data(2, 43, "/private/home/baharef/inversecooking2.0/data/flavorgraph")