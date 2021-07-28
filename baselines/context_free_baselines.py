import json
import pickle

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
    accuracy = 0
    for example in examples_cf:
        try:
            if example[1] in food2vec_subs_dict[example[0]]:
                accuracy += 1
        except Exception as e:
            # print(vocabs.idx2word[example[0]])
            pass
    return accuracy/len(examples_cf)

if __name__ == "__main__":
    for k in [5, 10, 1000]:
        for mode in [0, 1]:
            print('results with k ' + str(k) + ' and mode ' + str(mode))
            print("Accuracy on test:", test_food2vec(split='test', k=k, mode=mode)*100)
            print("Accuracy on val:", test_food2vec(split='val', k=k, mode=mode)*100)
            print("Accuracy on train:", test_food2vec(split='train', k=k, mode=mode)*100)
    
    