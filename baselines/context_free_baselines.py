import json
import pickle
import random

def context_free_examples(examples, mode, vocabs):
    output = []
    for example in examples:
        subs = example['subs']
        subs = vocabs.word2idx[subs[0]], vocabs.word2idx[subs[1]]
        if mode == 0:
            output.append(subs)
        elif mode == 1:
            if subs  not in output:
                output.append(subs)
    return output

def load_split_data(split):
    examples = json.load(open('../new/old/' + split + '_comments_subs.txt', 'r'))
    return examples

def load_dict(subs, vocabs):
    subs_dict = {}

    for ing in subs:
        ing_id = vocabs.word2idx[ing]
        subs_list = []
        for ing_subs in subs[ing]:
            subs_list.append(vocabs.word2idx[ing_subs.replace(' ', '_')])
        subs_dict[ing_id] = subs_list
    return subs_dict

def load_vocab():
    vocab_ing = pickle.load(open('../new/final_recipe1m_vocab_ingrs.pkl', 'rb'))
    return vocab_ing

def test_model(model_name, split='test', mode=0):
    vocabs = load_vocab()
    examples = load_split_data(split)
    examples_cf = context_free_examples(examples, mode, vocabs)
    subs = json.load(open('/private/home/baharef/temp/Exploiting-Food-Embeddings-for-Ingredient-Substitution/' + model_name + '_embeddings/data/substitute_pairs_' + model_name + '.json', 'r'))

    print("dictionary loaded!")
    
    subs_dict = load_dict(subs, vocabs)
    accuracy = 0
    mrr = 0
    hits = {1:0, 3:0, 10:0}
    for example in examples_cf:        
        try:
            subs = subs_dict[example[0]]
            rank = subs.index(example[1]) + 1
        except Exception as e:
            rank = random.randint(0, 6633) + 1
        mrr += (1/rank)
        if rank <= 1:
            hits[1] += 1
        if rank <= 3:
            hits[3] += 1
        if rank <= 10:
            hits[10] += 1
            
    for key in hits:
        hits[key] = hits[key] / len(examples_cf) * 100
    mrr = mrr/len(examples_cf) * 100
    return mrr, hits

if __name__ == "__main__":
    model_name = 'foodbert'
    mrr, hit = test_model(model_name, split='test', mode=0)
    print(mrr)
    print("Accuracy on test:", hit)
    

    