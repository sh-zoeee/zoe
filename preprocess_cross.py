from scripts import w_pq_batch as w_pq
import random

import numpy as np
from sklearn.model_selection import train_test_split
import pyconll
from pqgrams.PQGram import Profile
from scripts import pq_gram, trees, func
import torch
from torch import nn, optim
from torch.nn import DataParallel
from random import choices
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from time import time
from os import mkdir, makedirs, path


from statistics import mode


def main():
    random.seed(10)
    
    random_state = 10
    label_type = "unlabel"
    p = 2
    q = 2
    k_tree = 500

    torch.cuda.empty_cache()

    corpora = list()
    corpora.append(f"corpora/English/English-EWT.conllu")
    corpora.append(f"corpora/English/English-EWT.conllu")
    #corpora.append(f"parsed/En-EWT-udpipe.conllu")
    corpora.sort()
    CORPORA = corpora
    
    
    text = CORPORA[0].split("/")[-1].split(".")[0] + "_"
    dir_name = text
    text = CORPORA[1].split("/")[-1].split(".")[0]
    dir_name += text
    dir_name += f"_{2*k_tree}"
    print(dir_name)


    DIR_NAME = f"data_cross/{label_type}/" + dir_name + "/"
    if not path.exists(DIR_NAME):
        mkdir(DIR_NAME)
    
    #CORPORA =   [f"PUD/{source}-PUD.conllu", f"PUD/{target}-PUD.conllu"]
    LABELS = [0,1]

    CoNLL_src = random.sample(pyconll.load_from_file(CORPORA[0]), k=k_tree)
    CoNLL_tar = random.sample(pyconll.load_from_file(CORPORA[1]), k=k_tree)

    len_src = len(CoNLL_src)
    len_tar = len(CoNLL_tar)
    len_all = len_src + len_tar

    labels = []
    for _ in range(len_src):
        labels.append(0)
    for _ in range(len_tar):
        labels.append(1)
    
    CoNLL = CoNLL_src
    CoNLL += CoNLL_tar

    if label_type == "upos":
        pq_trees = [trees.conllTree_to_pqTree_upos(conll.to_tree()) for conll in CoNLL]
    elif label_type == "unlabel":
        pq_trees = [trees.conllTree_to_pqTree_unlabeled(conll.to_tree()) for conll in CoNLL]

    pqindex = [Profile(tree, p=p, q=q) for tree in pq_trees]

    J = set(pqindex[0])
    for pqset in pqindex[1:]:
        J = J.union(pqset)
    J = list(J)

    tensors = [pq_gram.pqgram_to_tensor(pqgram, J) for pqgram in tqdm(pqindex, desc="[convert tensor]")]

    indexes = torch.Tensor(range(len_all))

    fold_size = int(0.2*len_all)

    tensor1, _tensors, labels1, _labels, indexes1, _indexes = train_test_split(
        tensors, labels, indexes, train_size=fold_size, random_state=random_state
    )

    tensor2, _tensors, labels2, _labels, indexes2, _indexes = train_test_split(
        _tensors, _labels, _indexes, train_size=fold_size, random_state=random_state
    )

    tensor3, _tensors, labels3, _labels, indexes3, _indexes = train_test_split(
        _tensors, _labels, _indexes, train_size=fold_size, random_state=random_state
    )

    tensor4, tensor5, labels4, labels5, indexes4, indexes5 = train_test_split(
        _tensors, _labels, _indexes, train_size=fold_size, random_state=random_state
    )

    i = 1
    torch.save(tensor1, DIR_NAME+f"tensor_{i}.pt")
    torch.save(labels1, DIR_NAME+f"labels_{i}.pt")
    torch.save(indexes1, DIR_NAME+f"indexes_{i}.pt")

    i += 1
    torch.save(tensor2, DIR_NAME+f"tensor_{i}.pt")
    torch.save(labels2, DIR_NAME+f"labels_{i}.pt")
    torch.save(indexes2, DIR_NAME+f"indexes_{i}.pt")

    i += 1
    torch.save(tensor3, DIR_NAME+f"tensor_{i}.pt")
    torch.save(labels3, DIR_NAME+f"labels_{i}.pt")
    torch.save(indexes3, DIR_NAME+f"indexes_{i}.pt")

    i += 1
    torch.save(tensor4, DIR_NAME+f"tensor_{i}.pt")
    torch.save(labels4, DIR_NAME+f"labels_{i}.pt")
    torch.save(indexes4, DIR_NAME+f"indexes_{i}.pt")

    i += 1
    torch.save(tensor5, DIR_NAME+f"tensor_{i}.pt")
    torch.save(labels5, DIR_NAME+f"labels_{i}.pt")
    torch.save(indexes5, DIR_NAME+f"indexes_{i}.pt")


    

    
    

if __name__=="__main__":
    main()

