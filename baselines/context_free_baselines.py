import json

def context_free_examples(examples):
    output = []
    for example in examples:
        output.append(example['subs'])
    return output

def load_split_data(split):
    examples = json.load(open('../new/' + split + '_comments_subs.txt', 'r'))
    return examples

def test_food2vec(split='test', k=5):
    examples = load_split_data(split)
    examples_cf = context_free_examples(examples)
    food2vec_subs = json.load(open('/private/home/baharef/temp/Exploiting-Food-Embeddings-for-Ingredient-Substitution/food2vec/data/substitute_pairs_food2vec_text_Bahare_k'+ str(k) +'.json', 'r'))
    accuracy = 0
    for example in examples_cf:
        if list(example) in food2vec_subs:
            accuracy += 1
    return accuracy/len(examples_cf)

if __name__ == "__main__":
    k = 1000
    print("Accuracy on test:", test_food2vec(split='test', k=k))
    print("Accuracy on val:", test_food2vec(split='val', k=k))
    print("Accuracy on train:", test_food2vec(split='train', k=k))
    
    