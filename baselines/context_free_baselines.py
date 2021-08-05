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
    examples = json.load(open('../new/' + split + '_comments_subs.txt', 'r'))
    return examples

def load_food2vec_dict(food2vec_subs, vocabs):
    subs_dict = {}
    for subs in food2vec_subs:
        subs[0] = subs[0].replace(' ', '_')
        subs[1] = subs[1].replace(' ', '_')
        subs = vocabs.word2idx[subs[0]], vocabs.word2idx[subs[1]]
        if subs[0] not in subs_dict:
            subs_dict[subs[0]] = []
        if subs[1] not in subs_dict[subs[0]]:
            subs_dict[subs[0]].append(subs[1])
    if subs[0] == 'butter':
    # for key in subs_dict:
        print('butter', subs_dict['butter'])
    return subs_dict

def load_vocab():
    vocab_ing = pickle.load(open('../new/final_recipe1m_vocab_ingrs.pkl', 'rb'))
    return vocab_ing

def test_food2vec(split='test', k=5, mode=0):
    vocabs = load_vocab()
    examples = load_split_data(split)
    examples_cf = context_free_examples(examples, mode, vocabs)
    food2vec_subs = json.load(open('/private/home/baharef/temp/Exploiting-Food-Embeddings-for-Ingredient-Substitution/food2vec/data/substitute_pairs_food2vec_text_Bahare_k'+ str(k) +'.json', 'r'))
    food2vec_subs_dict = load_food2vec_dict(food2vec_subs, vocabs)

    food2vec_subs_dict = pickle.load(open('temp.pkl', 'rb'))
    print(food2vec_subs_dict[vocabs.word2idx['salt']][:10])
    exit()
    print("dictionary loaded!")
    accuracy = 0
    mrr = 0
    hits = {1:0, 3:0, 10:0}
    for example in examples_cf:
        
        
        print(food2vec_subs_dict[example[0]])
        print(example)
        exit()
        try:
            temp = food2vec_subs_dict[example[0]].reverse()
            rank = temp.index(example[1]) + 1
            print("Here")
            print(vocabs.idx2word[example[1]])
            print(vocabs.idx2word[food2vec_subs_dict[example[0]][0]])
            exit()
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
    for k in [6612]:
        for mode in [0]:
            print('results with k ' + str(k) + ' and mode ' + str(mode))
            mrr, hit = test_food2vec(split='test', k=k, mode=mode)
            print(mrr)
            print("Accuracy on test:", hit)
            # print("Accuracy on val:", test_food2vec(split='val', k=k, mode=mode)*100)
            # print("Accuracy on train:", test_food2vec(split='train', k=k, mode=mode)*100)
    
    