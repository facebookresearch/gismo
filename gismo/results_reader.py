import os
from collections import Counter
import numpy as np
   
def preprocess_line(line):
    new_line = line[3:]
    new_line = new_line.replace("[", "")
    new_line = new_line.replace("]", "")
    new_line = new_line.replace("{", "")
    new_line = new_line.replace("}", "")
    new_line = new_line.replace("1:", "")
    new_line = new_line.replace("3:", "")
    new_line = new_line.replace("10:", "")
    new_line = new_line.replace(",", "")
    # print(new_line)
    return new_line.split()


def preprocess_line2(line):
    new_line = line[3:].strip()
    new_line = new_line.replace("[", "")
    new_line = new_line.replace("]", "")
    new_line = new_line.replace("{", "")
    new_line = new_line.replace("}", "")
    new_line = new_line.replace("1:", "")
    new_line = new_line.replace("3:", "")
    new_line = new_line.replace("10:", "")
    new_line = new_line.replace(",", "")
    new_line = new_line.replace("device='cuda:0'", "")
    new_line = new_line.replace("dtype=torch.float64)", "")
    new_line = new_line.replace("tensor(", "")
    return new_line.split()


def LastNlines(fname, N):

    assert N >= 0
    pos = N + 1

    lines = []
    with open(fname, "r") as f:
        while len(lines) <= N:
            try:
                f.seek(-pos, 2)
            except IOError:
                f.seek(0)
                break
            finally:
                lines = list(f)
            pos *= 2
        f.close()
    return lines[-N:]


setup = "context-full"
val_only = True
name = "MLP_CAT"
dir_ = "/checkpoint/baharef/" + setup + "/" + name + "/oct-24/stdout/"
entries = os.listdir(dir_)

mrr_val = {}
mrr_test = {}
hit1_test = {}
hit3_test = {}
hit10_test = {}

failed_counter = 0

successed_counter = 0
for entry in entries:
    if not entry.startswith("."):
    #    if 'with_titles_False_with_set_False' in entry and 'lr_0.001_w_decay_0.0001_hidden_600_emb_d_600_dropout_0.25_nr_400_nlayers_2_lambda_0.0_i_' in entry:
        # if '_init_emb_random_walk_' in entry and 'lr_0.0005_w_decay_0.0001_hidden_500_emb_d_64_dropout_0.25_nr_400_nlayers_2_lambda_0.0_i_' in entry and '48438856' not in entry and '48438858' not in entry:
        # if 'bidir' in entry and 'lr_0.00005_w_decay_0.0001_hidden_400_emb_d_400_dropout_0.25_nr_400_nlayers_2_lambda_0.0_i_' in entry and 'p_augmentation_0.1' in entry:
        if 'with_titles_True_with_set_True' in entry and 'i_2' in entry:
            x = LastNlines(dir_ + entry, 6)
            # print(x)
            if val_only:
                try:
                    val_mrr, hit1, hit3, hit10 = preprocess_line2(x[3])
                    # print(val_mrr, hit1, hit3, hit10)
                    mrr_val[entry] = float(val_mrr)
                    hit1_test[entry] = float(hit1)
                    hit3_test[entry] = float(hit3)
                    hit10_test[entry] = float(hit10)
                    successed_counter += 1
                except:
                    failed_counter += 1
            else:
                try:
                    val_mrr, test_mrr, hit1, hit3, hit10 = preprocess_line2(x[-1])
                    mrr_val[entry] = float(val_mrr)
                    mrr_test[entry] = float(test_mrr)
                    hit1_test[entry] = float(hit1)
                    hit3_test[entry] = float(hit3)
                    hit10_test[entry] = float(hit10)
                    successed_counter += 1
                except:
                    try:
                        test_mrr, hit1, hit3, hit10 = preprocess_line(x[-1])
                        # mrr_val[entry] = float(val_mrr)
                        mrr_test[entry] = float(test_mrr)
                        hit1_test[entry] = float(hit1)
                        hit3_test[entry] = float(hit3)
                        hit10_test[entry] = float(hit10)
                        successed_counter += 1
                    except:
                        try:
                            test_mrr, hit1, hit3, hit10 = preprocess_line2(x[-1])
                            # mrr_val[entry] = float(val_mrr)
                            mrr_test[entry] = float(test_mrr)
                            hit1_test[entry] = float(hit1)
                            hit3_test[entry] = float(hit3)
                            hit10_test[entry] = float(hit10)
                            successed_counter += 1
                        except:
                            failed_counter += 1

print("failed", failed_counter)
print("successed", successed_counter)

if val_only:
    k = Counter(mrr_val)
    high = k.most_common(5)
    val_mrr = []
    hit1 = []
    hit3 = []
    hit10 = []
    for i in high:
        print(i[0], " :", i[1], " ")
        print(mrr_val[i[0]], hit1_test[i[0]], hit3_test[i[0]], hit10_test[i[0]])
        val_mrr.append(i[1])
        hit1.append(hit1_test[i[0]])
        hit3.append(hit3_test[i[0]])
        hit10.append(hit10_test[i[0]])
    print(np.mean(val_mrr), np.std(val_mrr))
    print(np.mean(hit1), np.std(hit1))
    print(np.mean(hit3), np.std(hit3))
    print(np.mean(hit10), np.std(hit10))

    print("val_mrrs:", val_mrr)
    print("hit1:", hit1)
    print("hit3:", hit3)
    print("hit10:", hit10)
else:
    k = Counter(mrr_val)
    high = k.most_common(5)

    val_mrr = []
    test_mrr = []
    hit1 = []
    hit3 = []
    hit10 = []
    for i in high:
        print(i[0], " :", i[1], " ")
        print(mrr_test[i[0]], hit1_test[i[0]], hit3_test[i[0]], hit10_test[i[0]])
        val_mrr.append(i[1])
        test_mrr.append(mrr_test[i[0]])
        hit1.append(hit1_test[i[0]])
        hit3.append(hit3_test[i[0]])
        hit10.append(hit10_test[i[0]])

    print(np.mean(val_mrr), np.std(val_mrr))
    print(np.mean(test_mrr), np.std(test_mrr))
    print(np.mean(hit1), np.std(hit1))
    print(np.mean(hit3), np.std(hit3))
    print(np.mean(hit10), np.std(hit10))

