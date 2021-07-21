import re
import os
import pickle
from typing import Any, Dict, List


class IngredientParser:
    def __init__(self, replace_dict: Dict[str, List[str]]):
        self.replace_dict = replace_dict

    def parse_entry(self, det_entry: Dict[str, Any], clean_digits=False):
        """
        Parse a detected ingredient entry and yield the valid ingredients it contains
        """
        det_ingrs = det_entry["ingredients"]
        valid = det_entry["valid"]
        for j, det_ingr in enumerate(det_ingrs):
            if len(det_ingr) > 0 and valid[j]:
                yield self.clean(det_ingr, clean_digits)

    def clean(self, det_ingr, clean_digits=False) -> str:
        """
        Read an ingredient ingredient and clean the ingredient
        - remove case
        - get rid of some special characters
        - replace some characters
        """
        det_ingr_undrs = det_ingr["text"].lower()
        if clean_digits:
            det_ingr_undrs = "".join(c for c in det_ingr_undrs if not c.isdigit())

        for rep, char_list in self.replace_dict.items():
            for c_ in char_list:
                if c_ in det_ingr_undrs:
                    det_ingr_undrs = det_ingr_undrs.replace(c_, rep)

        det_ingr_undrs = det_ingr_undrs.strip()
        det_ingr_undrs = det_ingr_undrs.replace(" ", "_")

        if not clean_digits:
            pattern = "(?P<char>[" + re.escape("_") + "])(?P=char)+"
            det_ingr_undrs = re.sub(pattern, r"\1", det_ingr_undrs)

        return det_ingr_undrs


def remove_plurals(counter_ingrs, ingr_clusters):
    deleted_ingredients = []

    for k, v in counter_ingrs.items():

        if len(k) == 0:
            deleted_ingredients.append(k)
            continue

        gotit = 0
        if k[-2:] == "es":
            if k[:-2] in counter_ingrs.keys():
                counter_ingrs[k[:-2]] += v
                ingr_clusters[k[:-2]].extend(ingr_clusters[k])
                deleted_ingredients.append(k)
                gotit = 1

        if k[-1] == "s" and gotit == 0:
            if k[:-1] in counter_ingrs.keys():
                counter_ingrs[k[:-1]] += v
                ingr_clusters[k[:-1]].extend(ingr_clusters[k])
                deleted_ingredients.append(k)
    for item in deleted_ingredients:
        del counter_ingrs[item]
        del ingr_clusters[item]
    return counter_ingrs, ingr_clusters


def cluster_ingredients(counter_ingrs):
    mydict = dict()
    mydict_ingrs = dict()

    for k, v in counter_ingrs.items():

        w1 = k.split("_")[-1]
        w2 = k.split("_")[0]
        lw = [w1, w2]
        if len(k.split("_")) > 1:
            w3 = k.split("_")[0] + "_" + k.split("_")[1]
            w4 = k.split("_")[-2] + "_" + k.split("_")[-1]

            lw = [w1, w2, w4, w3]

        gotit = 0
        for w in lw:
            if w in counter_ingrs.keys():
                # check if its parts are
                parts = w.split("_")
                if len(parts) > 0:
                    if parts[0] in counter_ingrs.keys():
                        w = parts[0]
                    elif parts[1] in counter_ingrs.keys():
                        w = parts[1]
                if w in mydict.keys():
                    mydict[w] += v
                    mydict_ingrs[w].append(k)
                else:
                    mydict[w] = v
                    mydict_ingrs[w] = [k]
                gotit = 1
                break
        if gotit == 0:
            mydict[k] = v
            mydict_ingrs[k] = [k]

    return mydict, mydict_ingrs


def remove_plurals_flavorgraph(counter_ingrs):
    deleted_ingredients = []
    ingr_clusters = {item: [item] for item in counter_ingrs.keys()}

    for ingr, count in counter_ingrs.items():

        if len(ingr) == 0:
            deleted_ingredients.append(ingr)
            continue

        gotit = 0
        key = ingr

        if (
            ingr[-3:] == "ves"
            and gotit == 0
            and ingr[:-3] + "f" in counter_ingrs.keys()
        ):
            key = ingr[:-3] + "f"
            counter_ingrs[key] += count
            ingr_clusters[key].extend(ingr_clusters[ingr])
            deleted_ingredients.append(ingr)
            gotit = 1

        elif ingr[-3:] == "ies" and gotit == 0:
            for k in [ingr[:-3] + "ie", ingr[:-3] + "i", ingr[:-3] + "y"]:
                if k in counter_ingrs.keys():
                    counter_ingrs[k] += count
                    ingr_clusters[k].extend(ingr_clusters[ingr])
                    deleted_ingredients.append(ingr)
                    gotit = 1
                    break

        if ingr[-2:] == "es" and gotit == 0 and ingr[:-2] in counter_ingrs.keys():
            key = ingr[:-2]
            counter_ingrs[key] += count
            ingr_clusters[key].extend(ingr_clusters[ingr])
            deleted_ingredients.append(ingr)
            gotit = 1

        if ingr[-1] == "s" and gotit == 0:
            for k in [ingr[:-1], ingr[:-1] + "es"]:
                if k in counter_ingrs.keys():
                    counter_ingrs[k] += count
                    ingr_clusters[k].extend(ingr_clusters[ingr])
                    deleted_ingredients.append(ingr)
                    gotit = 1
                    break

    for item in set(deleted_ingredients):
        del counter_ingrs[item]
        del ingr_clusters[item]

    return ingr_clusters, counter_ingrs


def match_flavorgraph(counter_ingrs, ingr_clusters, ingrs_flavorgraph, recipe1m_path):
    deleted_keys = []

    # difference betwenn ingredient clusters and ingrs in flavorgraph
    diff_ingrs = list(set(ingrs_flavorgraph).difference(set(ingr_clusters.keys())))

    for flavor_ingr in diff_ingrs[1:]:
        for k in ingr_clusters.keys():
            if flavor_ingr != k:
                check_list = [
                    flavor_ingr + "s",
                    flavor_ingr + "es",
                    flavor_ingr[:-1] + "ves",
                    flavor_ingr[:-3] + "f",
                    flavor_ingr[:-1] + "ve",
                    flavor_ingr[:-1] + "ies",
                    flavor_ingr[:-1] + "ie",
                ]
                # if flavor_ingr is within the cluster or is a version of the key, make sure it is used as key
                if flavor_ingr in ingr_clusters[k] or k in check_list:
                    ingr_clusters[flavor_ingr] = [flavor_ingr]
                    ingr_clusters[flavor_ingr].extend(ingr_clusters[k])
                    counter_ingrs[flavor_ingr] = counter_ingrs[k]
                    deleted_keys.append(k)
                    break

    # add missing flavor graph entries
    ingr_clusters["dried_tomato"] = ["dried_tomato"]
    ingr_clusters["dried_tomato"].extend(ingr_clusters["dried_tomatoe"])
    counter_ingrs["dried_tomato"] = counter_ingrs["dried_tomatoe"]
    deleted_keys.append("dried_tomatoe")
    ingr_clusters["rom"] = ["rom"]
    ingr_clusters["rom"].extend(ingr_clusters["roma"])
    counter_ingrs["rom"] = counter_ingrs["roma"]
    deleted_keys.append("roma")

    # merge redundant entries
    ingr_clusters["pimento"].extend(ingr_clusters["pimiento"])
    ingr_clusters["pimento"].extend(ingr_clusters["pimento_pepper"])
    counter_ingrs["pimento"] += counter_ingrs["pimiento"]
    counter_ingrs["pimento"] += counter_ingrs["pimento_pepper"]
    deleted_keys.append("pimiento")
    deleted_keys.append("pimento_pepper")

    for item in set(deleted_keys):
        del ingr_clusters[item]
        del counter_ingrs[item]

    # keep only ingredients in flavorgraph
    found_flavor = []
    for flavor_ingr in ingrs_flavorgraph:
        for items in ingr_clusters.items():
            if flavor_ingr in items[1]:
                found_flavor.append(items[0])

    # load the hand-crafted mapping from recipe1m ingredients to flavorgraph ingredients
    mapping = pickle.load(open(os.path.join(recipe1m_path, "merge_dict.pkl"), "rb"))
    for key in mapping:
        flavor_ing = mapping[key]
        if len(flavor_ing) > 0:
            if flavor_ing in found_flavor:
                ingr_clusters[flavor_ing].append(key)

    final_clusters = {k: ingr_clusters[k] for k in found_flavor}
    final_counters = {k: counter_ingrs[k] for k in found_flavor}
    return final_clusters, final_counters
