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
import sys
import multiprocessing as mp

import wasserTest_defs

def conll_to_zss_unlabel(tree: pyconll.tree)->zss.Node:
    node = zss.Node("_")
    for child in tree:
        node.addkid(conll_to_zss_unlabel(child))
    return node


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
    
    cpu_num = mp.cpu_count()
    pairs = []
    for i in range(numData):
        for j in range(i):
            pairs.append((setTreeN[i],setTreeN[j]))

    p = mp.Pool(cpu_num)            
    #result = p.map(calcDistOne, tqdm(pairs))
    result = p.map(calcDistOne, tqdm(pairs, leave=False))
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


def main():

    N = 20

    ewt = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-EWT"])
    #atis = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-Atis"])
    #esl = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-ESL"])
    cf = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-RandCF"])

    uniform = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-Uniform"])
    markov2 = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-Markov2"])
    markov3 = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-Markov3"])
    markov5 = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-Markov5"])
    markov10 = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-Markov10"])
    gpt = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-ChatGPT"])

    spacy = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-EWT-spacy"])
    udpipe = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-EWT-udpipe"])




    tree_ewt_N = []
    for conll in ewt:
        if len(conll) == N:
            tree_ewt_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_ewt_N = random.sample(tree_ewt_N, k=200)

    """
    tree_atis_N = []
    for conll in atis:
        if len(conll) == N:
            tree_atis_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    #tree_atis_N = random.sample(tree_atis_N, k=200)

    tree_esl_N = []
    for conll in esl:
        if len(conll) == N:
            tree_esl_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_esl_N = random.sample(tree_esl_N, k=200)
    """

    tree_cf_N = []
    for conll in cf:
        if len(conll) == N:
            tree_cf_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_cf_N = random.sample(tree_cf_N, k=200)

    tree_uniform_N = []
    for conll in uniform:
        if len(conll) == N:
            tree_uniform_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_uniform_N = random.sample(tree_uniform_N, k=200)

    tree_markov2_N = []
    for conll in markov2:
        if len(conll) == N:
            tree_markov2_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_markov2_N = random.sample(tree_markov2_N, k=200)

    tree_markov3_N = []
    for conll in markov3:
        if len(conll) == N:
            tree_markov3_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_markov3_N = random.sample(tree_markov3_N, k=200)

    tree_markov5_N = []
    for conll in markov5:
        if len(conll) == N:
            tree_markov5_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_markov5_N = random.sample(tree_markov5_N, k=200)

    tree_markov10_N = []
    for conll in markov10:
        if len(conll) == N:
            tree_markov10_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_markov10_N = random.sample(tree_markov10_N, k=200)

    tree_gpt_N = []
    for conll in gpt:
        if len(conll) == N:
            tree_gpt_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_gpt_N = random.sample(tree_gpt_N, k=200)


    tree_spacy_N = []
    for conll in spacy:
        if len(conll) == N:
            tree_spacy_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_spacy_N = random.sample(tree_spacy_N, k=200)

    tree_udpipe_N = []
    for conll in udpipe:
        if len(conll) == N:
            tree_udpipe_N.append(
                conll_to_zss_unlabel(conll.to_tree())
            )
    tree_udpipe_N = random.sample(tree_udpipe_N, k=200)


    distArray = calcDist(
        tree_ewt_N + 
        #tree_atis_N + 
        #tree_esl_N + 
        tree_cf_N + 
        tree_uniform_N +
        tree_markov2_N +
        tree_markov3_N +
        tree_markov5_N +
        tree_markov10_N +
        tree_gpt_N 
        #tree_spacy_N +
        #tree_udpipe_N
        )
    np.save(wasserTest_defs.DISTMX_DIR+"distmx_randsall.npy", distArray)

    return



def plot_multiple(distmx_path:str, output_path:str="multiple_tsne.png"):
    print(f"Plotting {distmx_path}...")

    distmx = np.load(distmx_path)
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(distmx)
    
    print(len(distmx))
    plt.clf()
    fig,ax = plt.subplots(tight_layout=True)
    ax.scatter(X_reduced[0:200, 0], X_reduced[0:200, 1], c='red', label='English-EWT')
    ax.scatter(X_reduced[200:400, 0], X_reduced[200:400, 1], c='blue', label='French-GSD')
    ax.scatter(X_reduced[400:600, 0], X_reduced[400:600, 1], c='green', label='Japanese-BCCWJ')
    ax.scatter(X_reduced[600:800, 0], X_reduced[600:800, 1], c='lightgreen', label='Korean-Kaist')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.savefig(output_path)

    print(f"Saved to {output_path}.")
    return


def plot_english(distmx_path:str, output_path:str="english_tsne.png"):
    print(f"Plotting {distmx_path}...")
    distmx = np.load(distmx_path)
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(distmx)
    
    print(len(distmx))
    plt.clf()
    fig,ax = plt.subplots(tight_layout=True)
    ax.scatter(X_reduced[0:200, 0], X_reduced[0:200, 1], c='red', label='English-EWT')
    ax.scatter(X_reduced[200:270, 0], X_reduced[200:270, 1], c='blue', label='English-Atis')
    ax.scatter(X_reduced[270:470, 0], X_reduced[270:470, 1], c='green', label='English-ESL')
    ax.scatter(X_reduced[470:670, 0], X_reduced[470:670, 1], c='orange', label='English-RandCF')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.savefig(output_path)

    print(f"Saved to {output_path}.")
    return




if __name__ == "__main__":
    main()
    #plot_multiple(wasserTest_defs.DISTMX_DIR+"distmx_multiple.npy", output_path="multiple_tsne.png")
    #plot_english(wasserTest_defs.DISTMX_DIR+"distmx_english.npy", output_path="english_tsne.png")

