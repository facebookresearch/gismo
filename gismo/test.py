import os
import torch
import pickle
from utils import load_foodbert, get_bert_emb
import faiss

def get_embedding(titles):
    tokenizer, bert = load_foodbert()
    device = torch.device('cuda:1')
    ing_embeddings_bert = get_bert_emb(titles, device, tokenizer, bert)
    return ing_embeddings_bert

def compute_embeddings():
    recipe_id2counter = pickle.load(open("/private/home/baharef/inversecooking2.0/proposed_model/titles_needed.pkl", "rb"))

    preprocessed_dir = "/private/home/baharef/inversecooking2.0/preprocessed_data"
    splits = ["train", "val", "test"]

    titles = []
    ids = []
    for split in splits:
        examples = pickle.load(open(os.path.join(preprocessed_dir, "final_recipe1m_" + split + ".pkl"), "rb"))
        for example in examples:
            title = ' '.join(example['title'])
            id = example['id']
            if id in recipe_id2counter:
                titles.append(title)
                ids.append(id)
    print(len(titles))
    return titles
    # titles_emb = get_embedding(titles)

    # pickle.dump(titles_emb, open("/private/home/baharef/inversecooking2.0/preprocessed_data/title_embeddings.pkl", "wb"))
    # pickle.dump(ids, open("/private/home/baharef/inversecooking2.0/preprocessed_data/title_recipe_ids.pkl", "wb"))

def load_embeddings():
    titles = compute_embeddings()
    embs = pickle.load(open('/private/home/baharef/inversecooking2.0/preprocessed_data/title_embeddings.pkl', 'rb'))
    cpu_index = faiss.IndexFlatL2(len(embs[0]))
    cpu_index.add(embs.numpy().astype("float32"))
    kth_values, kth_values_arg = cpu_index.search(embs.numpy().astype("float32"), k=3)

    for ind in [3, 5, 7]:
        print(titles[ind])
        for neigh in kth_values_arg:
            print(neigh)
            print(titles[neigh])
        print("*****")

    ind
if __name__ == "__main__":
    load_embeddings()