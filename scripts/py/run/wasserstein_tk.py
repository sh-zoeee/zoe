

from scripts import w_pq_batch as w_pq
from scripts import pq_gram, trees, func

import time, torch, random, pyconll, ot
import numpy as np

from os import path, mkdir

from tqdm.contrib import tenumerate

import matplotlib.pyplot as plt

from scripts.TreeKernel import tree, tree_kernels


def to_prolog_unlabel(tree: pyconll.tree.tree.Tree) -> str:

    if tree._children:
        children_repr = ', '.join(to_prolog_unlabel(child) for child in tree._children)
        return f'_({children_repr})'
    else:
        return f'_'
    

def to_prolog_upos(tree: pyconll.tree.tree.Tree) -> str:

    if tree._children:
        children_repr = ', '.join(to_prolog_upos(child) for child in tree._children)
        return f'{tree.data.upos}({children_repr})'
    else:
        return f'{tree.data.upos}'
    



def main():
    node_label = "upos"

    CoNLLU_source_PATH = "corpora/English/English-EWT.conllu"
    CoNLLU_target_PATH = "parsed/En-EWT-spacy.conllu"

    k = 5000
    CoNLL_src = random.sample(pyconll.load_from_file(CoNLLU_source_PATH), k=k)
    count_src = len(CoNLL_src)

    CoNLL_tar = random.sample(pyconll.load_from_file(CoNLLU_target_PATH), k=k)
    count_tar = len(CoNLL_tar)

    """
    if count_src==count_tar:
        labels = [0]*count_src + [1]*count_tar
    elif count_src < count_tar:
        CoNLL = CoNLL[:2*count_src]
        labels = [0]*count_src + [1]*count_src
    else:
        CoNLL = CoNLL[count_src-count_tar:]
        labels = [0]*count_tar + [1]*count_tar
    """
    labels = []
    for _ in range(count_src):
        labels.append(0)
    for _ in range(count_tar):
        labels.append(1)
    

    trees_src = [conll.to_tree() for conll in CoNLL_src]
    trees_tar = [conll.to_tree() for conll in CoNLL_tar]

    data_src = []
    data_tar = []

    source = CoNLLU_source_PATH.split("/")[-1].split(".")[0]
    text = source
    text += " vs "
    target = CoNLLU_target_PATH.split("/")[-1].split(".")[0].split("-")[-1] # 最後の.split()はparsedデータの場合に追加
    text += target

    LOG_DIR = f"logs/wasserstein/{node_label}/"
    LOG_FILE = f"tk_{source}_{target}.log"

    print("LOG -> ", LOG_DIR+LOG_FILE)

    if node_label == "unlabel":
        for t in trees_src:
            root = tree.TreeNode.fromPrologString(to_prolog_unlabel(t))
            data_src.append(tree.Tree(root))
        for t in trees_tar:
            root = tree.TreeNode.fromPrologString(to_prolog_unlabel(t))
            data_tar.append(tree.Tree(root))
    else:
        for t in trees_src:
            root = tree.TreeNode.fromPrologString(to_prolog_upos(t))
            data_src.append(tree.Tree(root))
        for t in trees_tar:
            root = tree.TreeNode.fromPrologString(to_prolog_upos(t))
            data_tar.append(tree.Tree(root))

    a = []
    for _ in range(count_src):
        a.append(1/count_src)
    b = []
    for _ in range(count_tar):
        b.append(1/count_tar)

    cost_matrix = torch.zeros((count_src, count_tar))

    lambda_value = 0.5
        
    kernel = tree_kernels.KernelST(l=lambda_value, savememory=0)

    kernel_self_src = [kernel.kernel(d,d) for d in data_src]
    kernel_self_tar = [kernel.kernel(d,d) for d in data_tar]

    for i, src in tenumerate(data_src, desc="[cost matrix]"):
        k1 = kernel_self_src[i]
        for j, tar in enumerate(data_tar):
            k2 = kernel_self_tar[j]
            k12 = kernel.kernel(src, tar)
            cost_matrix[i,j] = np.sqrt(k1+k2-2*k12)
            

    with open(LOG_DIR+LOG_FILE, mode="w") as f:
        print(text, "\n", file=f)
        print("wasserstein distance:", file=f)
        print(ot.emd2(a, b, cost_matrix.detach().numpy(), numItermax=1000000), file=f)


if __name__=="__main__":
    main()

