import json
import os
import pickle
import warnings
from pathlib import Path

# import faiss
import matplotlib.pyplot as plt
import torch
from hydra.experimental import compose, initialize
from sklearn.manifold import TSNE
from transformers import (
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    logging,
)

from inv_cooking.scheduler import RawConfig
from inv_cooking.training.trainer import load_data_set

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


class Bert:
    def __init__(self, device, tokenizer=None, bert=None):
        self.device = device
        if tokenizer:
            self.tokenizer = tokenizer
            print("tokenizer loaded")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            print("tokenizer loaded 2")
        if bert:
            # self.bert = bert.to(device=self.device)
            self.bert = bert
            print("bert loaded")
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased").to(
                device=self.device
            )
            print("bert loaded 2")
        self.bert.eval()
        print("BERT successfully loaded.")
    def get_contexts_for_one(self, batch_sentences):
        encoded_inputs = self.tokenizer(
            batch_sentences, padding=True, truncation=True, return_tensors="pt"
        )
        # outputs = self.bert(encoded_inputs["input_ids"].to(device=self.device))
        outputs = self.bert(encoded_inputs["input_ids"])
        return outputs

    def get_cls_emb_for_one(self, text):
        embs = self.get_contexts_for_one(text).last_hidden_state
        return embs[:, 0, :]

    def get_avg_emb_for_one(self, text):
        embs = self.get_contexts_for_one(text).last_hidden_state
        return torch.mean(embs[:, 1:-1, :], dim=1)


def load_config():
    with initialize(config_path="../conf"):
        cfg_init = compose(
            config_name="config",
            overrides=[
                "task=ingrsubs",
                "checkpoint=baharef",
                "dataset.loading.batch_size=1",
                "dataset.pre_processing.save_path=/private/home/baharef/inversecooking2.0/preprocessed_data",
            ],
        )
        configurations = RawConfig.to_config(cfg_init)
    return configurations[0]


def get_vocabs():
    cfg = load_config()
    data_module = load_data_set(cfg)
    data_module.prepare_data()
    data_module.setup("fit")
    vocab_ing = data_module.dataset_train.ingr_vocab
    vocab_title = data_module.dataset_train.title_vocab
    return vocab_ing, vocab_title


def load_data():
    cfg = load_config()
    data_module = load_data_set(cfg)
    data_module.prepare_data()
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    return train_dataloader, val_dataloader, test_dataloader, data_module


def ids_to_words(list_ing, vocab):
    res = []
    for ing in list_ing:
        ing_word = vocab.idx2word[ing]
        if ing_word not in ["<end>", "<pad>"]:
            res.append(ing_word)
    return res


def load_preprocessed_ingredients(dir):
    ingredients = pickle.load(
        open(os.path.join(dir, "ingredients_preprocessed.txt"), "rb")
    )
    return ingredients


def load_preprocessed_titles(dir):
    titles = pickle.load(open(os.path.join(dir, "titles_preprocessed.txt"), "rb"))
    return titles


def load_preprocessed_ids(dir):
    ids = pickle.load(open(os.path.join(dir, "ids_preprocessed.txt"), "rb"))
    return ids


def read_title_ingredients_ids(train_dataloader, vocab_ing, vocab_title):
    ingredients = []
    title = []
    ids = []
    counter = 0
    for batch in train_dataloader:
        if counter % 10000 == 0:
            print(counter)
        curr_title = batch["title"][0]
        curr_ingredients = batch["ingredients"][0]

        title.append(" ".join(ids_to_words(curr_title.numpy(), vocab_title)))
        ingredients.append(
            ing_preprocessing(ids_to_words(curr_ingredients.numpy(), vocab_ing))
        )
        ids.append(batch["id"][0])

        counter += 1

    return title, ingredients, ids


def read_titles_dict():
    train_dataloader, val_dataloader, test_dataloader, data_module = load_data()
    vocab_ing, vocab_title = get_vocabs(data_module)
    titles = {}
    counter = 0
    # for batch in train_dataloader:
    #     if counter % 10000 == 0:
    #         print(counter)
    #     curr_id = batch['id']
    #     curr_title = batch['title'][0]
    #     titles[curr_id]= ' '.join(ids_to_words(curr_title.numpy(), vocab_title))

    for batch in val_dataloader:
        if counter % 10000 == 0:
            print(counter)
        curr_id = batch['id'][0]
        curr_title = batch['title'][0]
        titles[curr_id]= ' '.join(ids_to_words(curr_title.numpy(), vocab_title))

    # for batch in test_dataloader:
    #     if counter % 10000 == 0:
    #         print(counter)
    #     curr_id = batch["id"]
    #     curr_title = batch["title"][0]
    #     titles[curr_id] = " ".join(ids_to_words(curr_title.numpy(), vocab_title))
    return titles


def ing_preprocessing(ings):
    res = []
    for ing in ings:
        res.append(ing[0])
    return res


def knn(X, k):
    cpu_index = faiss.IndexFlatL2(X.shape[1])
    cpu_index.add(X.cpu().float().numpy().astype("float32"))
    kth_values, kth_values_arg = cpu_index.search(X.cpu().numpy().astype("float32"), k)
    return kth_values, kth_values_arg


def get_bert_emb(titles, device, tokenizer=None, bert=None, batch_size=2000):
    print("Here is to compute the embedding")
    model = Bert(device, tokenizer, bert)
    index = 0
    embs = torch.zeros(len(titles), 768)
    # .to(device)
    with torch.no_grad():
        while index + batch_size < len(titles):
            print(index)
            embs[index : index + batch_size, :] = model.get_avg_emb_for_one(
                titles[index : index + batch_size]
            )
            index += batch_size
        embs[index : len(titles), :] = model.get_avg_emb_for_one(
            titles[index : len(titles)]
        )
    return embs


def get_synonyms(ing, vocab_ing):
    if ing in vocab_ing.word2idx:
        return vocab_ing.idx2word[vocab_ing.word2idx[ing]]
    else:
        return [ing]


# def make_consistent(ings1, ings2, vocab_ing):
#     for ing in ings1:
#         if ing not in ings2:
#             synonyms = get_synonyms(ing, vocab_ing)
#             syn_not_found = True
#             for syn in synonyms:
#                 if syn in ings2 and syn_not_found:
#                     ings2.remove(syn)
#                     ings2.append(ing)
#                     syn_not_found = False
#     return ings1, ings2


def compute_precision(xq_ing, neigh_ing):
    precision = 0.0
    n_p = 0.0
    for ing in neigh_ing:
        if ing in xq_ing:
            precision += 1.0
        n_p += 1.0
    return precision / n_p


def compute_recall(xq_ing, neigh_ing):
    recall = 0.0
    n_r = 0.0
    for ing in xq_ing:
        if ing in neigh_ing:
            recall += 1.0
        n_r += 1.0
    return recall / n_r


def compute_f1_score(xq_ing, neigh_ing):
    precision = compute_precision(xq_ing, neigh_ing)
    recall = compute_recall(xq_ing, neigh_ing)
    f1 = 2 * (precision * recall) / (precision + recall + 0.000001)
    return f1


def compute_number_of_different_ingredients(xq_ing, neigh_ing):
    return len(set(neigh_ing) - set(xq_ing))


def intersection(lst1, lst2):
    [value for value in lst1 if value in lst2]
    a = set(lst1)
    b = set(lst2)
    return a.intersection(b)


def union(lst1, lst2):
    return list(set(lst1) | set(lst2))


def IOU(lst1, lst2):
    return len(intersection(lst1, lst2)) / len(union(lst1, lst2))


def load_classes():
    with open("../data/recipe1m_processed/classes1M.pkl", "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def load_foodbert():

    # import sys

    # default_dir = (
    #     "/private/home/baharef/Exploiting-Food-Embeddings-for-Ingredient-Substitution/"
    # )
    # sys.path.insert(1, default_dir)
    # from foodbert.run_language_modeling import ModelArguments

    # model_args = ModelArguments(
    #     model_type="bert",
    #     model_name_or_path=default_dir + "foodbert/data/mlm_output4/",
    # )
    # config = AutoConfig.from_pretrained(
    #     model_args.model_name_or_path, cache_dir=model_args.cache_dir
    # )

    # # with Path(default_dir + "foodbert/data/used_ingredients.json").open() as f:
    # with Path('/private/home/baharef/inversecooking2.0/substitution_generation/used_ingredients_clean.json').open() as f:
    #     used_ingredients = json.load(f)  # Dont seperate these
    # tokenizer = BertTokenizer(
    #     vocab_file=default_dir + "foodbert/data/bert-base-cased-vocab.txt",
    #     do_lower_case=False,
    #     max_len=128,
    #     never_split=used_ingredients,
    # )  # For one sentence instruction, longer shouldn't be necessary
    # model = AutoModelWithLMHead.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    # )
    # return tokenizer, model.bert

    from transformers import BertModel, BertTokenizer
    model = BertModel.from_pretrained(pretrained_model_name_or_path='/private/home/baharef/Exploiting-Food-Embeddings-for-Ingredient-Substitution/foodbert/data/mlm_output4/')
    # pretrained_model_name_or_path='foodbert/data/mlm_output/checkpoint-final')
    # with open('foodbert/data/used_ingredients.json', 'r') as f:
    with open('/private/home/baharef/inversecooking2.0/substitution_generation/used_ingredients_clean.json', 'r') as f:
        used_ingredients = json.load(f)
    tokenizer = BertTokenizer(vocab_file='/private/home/baharef/Exploiting-Food-Embeddings-for-Ingredient-Substitution/foodbert/data/bert-base-cased-vocab.txt', do_lower_case=False,
                                    max_len=128, never_split=used_ingredients)


    return tokenizer, model

    


def get_ingredient_text(ingredients):
    ings_text = []
    for key in range(len(ingredients)):
        for ing in ingredients[key]:
            if ing not in ings_text:
                ings_text.append(ing)
    return ings_text


def many_to_one(set1, set2, ing_emb, ing_text, threshold=1.0):
    # match each ing in set1 to closes in set2 (many can be matched to one)
    if len(set1) == 0 or len(set2) == 0:
        return []
    res = []
    for ind1 in set1:
        distances = {}
        ind1_emb = ing_emb[ing_text.index(ind1)]
        for ind2 in set2:
            ind2_emb = ing_emb[ing_text.index(ind2)]
            d = torch.sum((ind1_emb - ind2_emb) ** 2)
            distances[ind2] = d.cpu().item()
        best_match = min(distances, key=distances.get)
        if distances[best_match] < threshold:
            res.append((ind1, best_match))
    return res


def one_to_one(set1, set2, ing_emb, ing_text, threshold=1.0):
    distances = {}
    for ind1 in set1:
        ind1_emb = ing_emb[ing_text.index(ind1)]
        for ind2 in set2:
            ind2_emb = ing_emb[ing_text.index(ind2)]
            d = torch.sum((ind1_emb - ind2_emb) ** 2)
            distances[(ind1, ind2)] = d.cpu().item()
    sorted_ = dict(sorted(distances.items(), key=lambda item: item[1]))
    res = []
    for pair in sorted_:
        if sorted_[pair] < threshold and pair[0] in set1 and pair[1] in set2:
            res.append((pair[0], pair[1]))
            set1.remove(pair[0])
            set2.remove(pair[1])
    return res


def many_to_many(set1, set2, ing_emb, ing_text, threshold=1.0):
    distances = {}
    for ind1 in set1:
        ind1_emb = ing_emb[ing_text.index(ind1)]
        for ind2 in set2:
            ind2_emb = ing_emb[ing_text.index(ind2)]
            d = torch.sum((ind1_emb - ind2_emb) ** 2)
            distances[(ind1, ind2)] = d.cpu().item()
    sorted_ = dict(sorted(distances.items(), key=lambda item: item[1]))
    res = []
    for pair in sorted_:
        if sorted_[pair] < threshold:
            res.append((pair[0], pair[1]))

    return res


def clean_substitue(x, y):
    x_, y_ = x.copy(), y.copy()
    for x_val in x:
        for y_val in y:
            if x_val in y_val or y_val in x_val:
                if x_val in x_:
                    x_.remove(x_val)
                if y_val in y_:
                    y_.remove(y_val)
    return x_, y_


def get_set_emb(set_, ing_emb, ing_text, device):
    res_emb = torch.zeros(len(set_), 768).to(device)
    for i in range(len(set_)):
        res_emb[i, :] = ing_emb[ing_text.index(set_[i])]
    return res_emb


def visualize_emb(ing_emb, ing_text):
    X_embedded = TSNE(n_components=2).fit_transform(ing_emb.cpu())
    for ind, ing in enumerate(ing_text):
        if "bell_pepper" in ing:
            # plt.annotate(ing, (X_embedded[ind, 0], X_embedded[ind, 1]))
            plt.scatter(X_embedded[ind, 0], X_embedded[ind, 1], c="green")
        elif "green_onion" in ing:
            plt.scatter(X_embedded[ind, 0], X_embedded[ind, 1], c="blue")
        elif "peanut" in ing:
            plt.scatter(X_embedded[ind, 0], X_embedded[ind, 1], c="red")
        elif "flour" in ing:
            plt.scatter(X_embedded[ind, 0], X_embedded[ind, 1], c="black")
        elif "milk" in ing:
            plt.scatter(X_embedded[ind, 0], X_embedded[ind, 1], c="purple")
    plt.show()


def get_category(xq_ind, class_dict, ind2class):
    class_ = class_dict[counter_to_id[xq_ind]]
    class_label = ind2class[class_]
    return class_, class_label
