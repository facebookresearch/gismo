import os
import shutil
from collections import deque

import hydra
from omegaconf import DictConfig, OmegaConf


def copy_source_code_to_cwd():
    """
    Copy the important folder to the path in which hydra will be running the code
    * this allows relative path to continue to work
    * this allows submitit to find the main module
    """
    original_path = hydra.utils.get_original_cwd()
    target_path = os.getcwd()
    folders_to_copy = ["inv_cooking", "data"]
    print(f"Copying {folders_to_copy} from {original_path} to {target_path}...")
    for folder in folders_to_copy:
        src_folder = os.path.join(original_path, folder)
        dst_folder = os.path.join(target_path, folder)
        shutil.copytree(src_folder, dst_folder, symlinks=True)
    print(f"Copying done. Running code in folder {target_path}")


def merge_with_non_missing(base: DictConfig, replacement: DictConfig):
    """
    A version of OmegaConf.merge which does not merge with MISSING values
    """
    merged = base.copy()
    to_visit = deque(replacement.keys())
    while to_visit:
        path = to_visit.popleft()
        node = OmegaConf.select(replacement, path)
        if OmegaConf.is_dict(node):
            for key in node.keys():
                to_visit.append(path + "." + key)
        elif OmegaConf.is_list(node):
            for i in range(len(node)):
                to_visit.append(path + "." + str(i))
        elif node is not None:
            OmegaConf.update(merged, path, node, merge=False)
    return merged
