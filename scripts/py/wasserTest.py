import random, copy, os
from graphviz import Graph
import matplotlib.pyplot as plt
import cv2

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

import zss
from zss import simple_distance
import pyconll
import ot

from tqdm import tqdm
import time
import sys
import multiprocessing as mp
import glob
from itertools import combinations

import wasserTest_defs


def time_to_hhmmss(seconds: int):
    hours = seconds // 3600
    minutes = (seconds%3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}:{int(minutes)}:{int(seconds)}"

def conll_to_zss_unlabel(tree: pyconll.tree)->zss.Node:
    node = zss.Node("_")
    for child in tree:
        node.addkid(conll_to_zss_unlabel(child))
    return node


def conll_to_zss_signed_ordered(tree: pyconll.tree) -> zss.Node:
    """Convert a pyconll tree to a signed ordered tree for ZSS.

    Each node label encodes the relative position of the node to its parent:
    - '-1': the node is to the left of its parent in the sentence order
    - '1' : the node is to the right of its parent in the sentence order
    - '0' : the root node

    Children are ordered by their token index to obtain an ordered tree.
    """

    def to_int_id(token_id) -> int:
        # Convert CoNLL-U token id to int (handles cases like '3', '3.1', '3-4')
        s = str(token_id)
        base = s.split('-')[0].split('.')[0]
        return int(base) if base.isdigit() else 10**9

    def build(node_tree, parent_idx: int | None) -> zss.Node:
        # Determine current node index
        cur_idx = to_int_id(node_tree.token.id) if hasattr(node_tree, 'token') else None

        # Label based on relative position to parent
        if parent_idx is None or cur_idx is None:
            label = '0'
        else:
            label = '1' if cur_idx > parent_idx else '-1'

        node = zss.Node(label)

        # Order children by token index to make it an ordered tree
        children = list(node_tree)
        children.sort(key=lambda c: to_int_id(c.token.id) if hasattr(c, 'token') else 10**9)

        for child in children:
            node.addkid(build(child, cur_idx))

        return node

    return build(tree, None)


def wasserstein_test(src_all, tgt_all, random_state=0, sample_size=200):
    
    random.seed(random_state)
    if len(src_all)>sample_size:
        src = random.sample(src_all, k=sample_size)
    else:
        src = copy.deepcopy(src_all)
    if len(tgt_all)>sample_size:
        tgt = random.sample(tgt_all, k=sample_size)
    else:
        tgt = copy.deepcopy(tgt_all)


    len_src = len(src)
    len_tgt = len(tgt)

    distArray = calcDist(src + tgt)
    costMatrix = cutOutInds(distArray, list(range(len_src)), list(range(len_src, len_src+len_tgt)))
    # Compute the Wasserstein distance

    weight_i = np.ones(len_src)/float(len_src)  #weight is equal        
    weight_j = np.ones(len_tgt)/float(len_tgt)  #weight is equal

    ot_emd = ot.emd(weight_i, weight_j, costMatrix)
    cost = np.sum(ot_emd * costMatrix)
    
    return distArray, cost


def cutOutInds(distArray,Xinds,Yinds):

    len_x = len(Xinds)
    len_y = len(Yinds)     
    
    matC = np.zeros((len_x, len_y), dtype=float)
    
    for i in range(len_x):
        for j in range(len_y):
            matC[i][j] = float(distArray[Xinds[i]][Yinds[j]])
            
    return matC


def calcDistOne(treepair):
    return float(simple_distance(treepair[0], treepair[1]))


def calcDist(setTreeN):
    numData = len(setTreeN)
    #print(".....calculating distances among "+str(numData)+" trees")
    
    distArray = np.zeros((numData, numData), dtype=float)
    
    cpu_num = 16
    pairs = []
    for i in range(numData):
        for j in range(i):
            pairs.append((setTreeN[i],setTreeN[j]))

    p = mp.Pool(cpu_num)            
    result = p.map(calcDistOne, pairs)
    #result = p.map(calcDistOne, tqdm(pairs, leave=False))
    #print("len=",len(result))
    
    count = 0
    for i in range(numData):
        distArray[i][i] = 0
        for j in range(i):
            distArray[i][j] = result[count]
            distArray[j][i] = result[count]
            count += 1
    p.close()

    return distArray


def main(src: str, tgt: str):

    N = 20

    corpora = wasserTest_defs.CORPORA_PATH.keys()
    if src not in corpora or tgt not in corpora:
        print(f"Invalid source or target language.")
        print(f"Available languages:")
        for cor in corpora:
            print(f"\t{cor}")
        print(f"\nPlease choose from the above list.\n")
        return
    path_src = wasserTest_defs.CORPORA_PATH[src]
    path_tgt = wasserTest_defs.CORPORA_PATH[tgt]
    
    conll_src = pyconll.load_from_file(path_src)
    conll_tgt = pyconll.load_from_file(path_tgt)

    print(f"source: {src}", flush=True)
    tree_src_N_all = []
    for conll in conll_src:
        if len(conll) == N:
            tree_src_N_all.append(conll.to_tree())
    tree_src_zss_N_all = [conll_to_zss_signed_ordered(tree) for tree in tree_src_N_all]
    print(f"\ttotal: {len(conll_src)}, N={N}: {len(tree_src_zss_N_all)}", flush=True)


    print(f"target: {tgt}", flush=True)
    tree_tgt_N_all = []
    for conll in conll_tgt:
        if len(conll) == N:
            tree_tgt_N_all.append(conll.to_tree())
    tree_tgt_zss_N_all = [conll_to_zss_signed_ordered(tree) for tree in tree_tgt_N_all]
    print(f"\ttotal: {len(conll_tgt)}, N={N}: {len(tree_tgt_zss_N_all)}", flush=True)

    cost_list = []
    for seed in range(10):
        print(f"STEP {1+seed}...", end="", flush=True)
        random.seed(seed)
        #print("seed: "+str(seed))
        #print("src: "+str(len(tree_src_zss_N_all)))
        #print("tgt: "+str(len(tree_tgt_zss_N_all)))
        #print("sample size: "+str(sample_size))
        
        distArray, cost = wasserstein_test(tree_src_zss_N_all, tree_tgt_zss_N_all, random_state=seed)
        cost_list.append(cost)
        print("\tdone.")
        np.save(wasserTest_defs.DISTMX_DIR+src+"_"+tgt+"_"+str(seed)+".npy", distArray)
    
    LOG_PATH = wasserTest_defs.LOG_DIR+src+"_"+tgt+".log"
    print("saving to "+LOG_PATH)
    with open(LOG_PATH, "w") as f:
        print(f"{np.mean(cost_list):.3f}$\pm${np.std(cost_list):.3f}", file=f)
        print("======================", file=f)
        print(f"{cost_list}", file=f)


if __name__ == "__main__":

    corpora_path = list(wasserTest_defs.CORPORA_PATH.keys())
    n = len(corpora_path)
    src = "English-EWT"
    
    """    
    start_time = time.time()
    for i, tgt in enumerate(corpora_path):

        print("########################################################")
        print(f"{tgt}\t({i+1}/{n})")
        main(src, tgt)

        elapsed_time = time.time() - start_time
        progress = (i+1)/n
        eta = (elapsed_time/progress) * (1-progress)
        print(f"\nETA: {time_to_hhmmss(eta)}s")
    """

    main(src, "English-EWT-gpt5")