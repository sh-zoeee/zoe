

from scripts import w_pq_batch as w_pq
from scripts import pq_gram, trees, func

import time, torch, random, pyconll, ot
import numpy as np

from pqgrams.PQGram import Profile
import zss

from tqdm.contrib import tenumerate

import matplotlib.pyplot as plt


def main():
    
    node_label = "unlabel"
    
    CoNLLU_source_PATH = "corpora/English/English-EWT.conllu"
    CoNLLU_src = random.sample(pyconll.load_from_file(CoNLLU_source_PATH), k=500)
    source_tree_count = len(CoNLLU_src)

    CoNLLU_target_PATH = f"parsed/En-EWT-udpipe.conllu"
    CoNLLU_tar = random.sample(pyconll.load_from_file(CoNLLU_target_PATH), k=500)
    target_tree_count = len(CoNLLU_tar)

    if node_label == "unlabel":
        zss_trees_src = [trees.conllTree_to_zssNode_unlabel(conll.to_tree()) for conll in CoNLLU_src]
        zss_trees_tar = [trees.conllTree_to_zssNode_unlabel(conll.to_tree()) for conll in CoNLLU_tar]
    else :
        zss_trees_src = [trees.conllTree_to_zssNode_upos(conll.to_tree()) for conll in CoNLLU_src]
        zss_trees_tar = [trees.conllTree_to_zssNode_upos(conll.to_tree()) for conll in CoNLLU_tar]


    a = []
    for _ in range(source_tree_count):
        a.append(1/source_tree_count)
    b = []
    for _ in range(target_tree_count):
        b.append(1/target_tree_count)

    cost_matrix = torch.zeros((source_tree_count, target_tree_count))

    source = CoNLLU_source_PATH.split("/")[-1].split(".")[0]
    text = source
    text += " vs "
    target = CoNLLU_target_PATH.split("/")[-1].split(".")[0].split("-")[-1]
    text += target

    print("LOG -> ", f"logs/wasserstein/{node_label}/ted_{source}_{target}.log")

    for i, src in tenumerate(zss_trees_src, desc="[cost matrix]"):
        for j, tar in enumerate(zss_trees_tar):
            cost_matrix[i,j] = zss.simple_distance(src, tar)

    with open(f"logs/wasserstein/{node_label}/ted_{source}_{target}.log", mode="w") as f:
        print(text, "\n", file=f)
        print("wasserstein distance:", file=f)
        print(ot.emd2(a, b, cost_matrix.detach().numpy(), numItermax=1000000), file=f)
    
    


if __name__=="__main__":
    main()

