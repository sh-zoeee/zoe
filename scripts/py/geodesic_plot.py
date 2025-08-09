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


def gen_distmx_english():

    #N = 20
    K = 200

    ewt = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-EWT"])
    atis = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-Atis"])
    esl = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-ESL"])
    rand = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-English-EWT-CF"])


    ewt = [conll for conll in ewt if len(conll) > 5]  # Filter out too short sentences
    atis = [conll for conll in atis if len(conll) > 5]  
    esl = [conll for conll in esl if len(conll) > 5]
    rand = [conll for conll in rand if len(conll) > 5]

    ewt = random.sample(ewt, k=K)
    count_ewt = [len(conll) for conll in ewt]

    atis = random.sample(atis, k=K)
    count_atis = [len(conll) for conll in atis]

    esl = random.sample(esl, k=K)
    count_esl = [len(conll) for conll in esl]


    rand = random.sample(rand, k=K)
    count_rand = [len(conll) for conll in rand]

    with open(wasserTest_defs.LOG_DIR+"english_counts_over5.csv", "w") as f:
        f.write("id, type, tree size\n")
        id = 0
        for num in count_ewt:
            f.write(f"{id}, EWT, {num}\n")
            id += 1
        for num in count_atis:
            f.write(f"{id}, Atis, {num}\n")
            id += 1
        for num in count_esl:
            f.write(f"{id}, ESL, {num}\n")
            id += 1
        for num in count_rand:
            f.write(f"{id}, RandCF, {num}\n")
            id += 1

    tree_ewt = [conll_to_zss_unlabel(conll.to_tree()) for conll in ewt]
    tree_atis = [conll_to_zss_unlabel(conll.to_tree()) for conll in atis]
    tree_esl = [conll_to_zss_unlabel(conll.to_tree()) for conll in esl]
    tree_rand = [conll_to_zss_unlabel(conll.to_tree()) for conll in rand]


    distArray = calcDist(tree_ewt + tree_atis + tree_esl + tree_rand)
    np.save(wasserTest_defs.DISTMX_DIR+"english_over5.npy", distArray)

    return


def gen_distmx_multiple():

    #N = 20
    K = 200

    ewt = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-EWT"])
    fr= pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["French-FTB"])
    ja = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Japanese-BCCWJ"])
    ko = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Korean-Kaist"])


    ewt = [conll for conll in ewt if len(conll) > 5]  # Filter out too short sentences
    fr = [conll for conll in fr if len(conll) > 5]  
    ja = [conll for conll in ja if len(conll) > 5]
    ko = [conll for conll in ko if len(conll) > 5]

    ewt = random.sample(ewt, k=K)
    count_ewt = [len(conll) for conll in ewt]

    fr = random.sample(fr, k=K)
    count_fr = [len(conll) for conll in fr]

    ja = random.sample(ja, k=K)
    count_ja = [len(conll) for conll in ja]


    ko = random.sample(ko, k=K)
    count_ko = [len(conll) for conll in ko]

    with open(wasserTest_defs.LOG_DIR+"multiple_counts_over5.csv", "w") as f:
        f.write("id, type, tree size\n")
        id = 0
        for num in count_ewt:
            f.write(f"{id}, English-EWT, {num}\n")
            id += 1
        for num in count_fr:
            f.write(f"{id}, French-GSD, {num}\n")
            id += 1
        for num in count_ja:
            f.write(f"{id}, Japanese-BCCWJ, {num}\n")
            id += 1
        for num in count_ko:
            f.write(f"{id}, Korean-Kaist, {num}\n")
            id += 1

    tree_ewt = [conll_to_zss_unlabel(conll.to_tree()) for conll in ewt]
    tree_fr = [conll_to_zss_unlabel(conll.to_tree()) for conll in fr]
    tree_ja = [conll_to_zss_unlabel(conll.to_tree()) for conll in ja]
    tree_ko = [conll_to_zss_unlabel(conll.to_tree()) for conll in ko]


    distArray = calcDist(tree_ewt + tree_fr + tree_ja + tree_ko)
    np.save(wasserTest_defs.DISTMX_DIR+"multiple_over5.npy", distArray)

    return



def gen_distmx_randsAll():

    #N = 20
    K = 200

    ewt = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-EWT"])
    rand = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-English-EWT-CF"])
    uniform = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-Uniform"])
    markov2 = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-Markov2"])
    markov3 = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-Markov3"])
    markov5 = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-Markov5"])
    markov10 = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-Markov10"])
    chatgpt = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-ChatGPT"])



    ewt = [conll for conll in ewt if len(conll) > 5]  # Filter out too short sentences
    rand = [conll for conll in rand if len(conll) > 5]
    uniform = [conll for conll in uniform if len(conll) > 5]
    markov2 = [conll for conll in markov2 if len(conll) > 5]
    markov3 = [conll for conll in markov3 if len(conll) > 5]
    markov5 = [conll for conll in markov5 if len(conll) > 5]
    markov10 = [conll for conll in markov10 if len(conll) > 5]
    chatgpt = [conll for conll in chatgpt if len(conll) > 5]


    ewt = random.sample(ewt, k=K)
    count_ewt = [len(conll) for conll in ewt]

    rand = random.sample(rand, k=K)
    count_rand = [len(conll) for conll in rand]

    uniform = random.sample(uniform, k=K)
    count_uniform = [len(conll) for conll in uniform]

    markov2 = random.sample(markov2, k=K)
    count_markov2 = [len(conll) for conll in markov2]
    markov3 = random.sample(markov3, k=K)
    count_markov3 = [len(conll) for conll in markov3]
    markov5 = random.sample(markov5, k=K)
    count_markov5 = [len(conll) for conll in markov5]
    markov10 = random.sample(markov10, k=K)
    count_markov10 = [len(conll) for conll in markov10]

    chatgpt = random.sample(chatgpt, k=K)
    count_chatgpt = [len(conll) for conll in chatgpt]


    with open(wasserTest_defs.LOG_DIR+"randsAll_counts_over5.csv", "w") as f:
        f.write("id, type, tree size\n")
        id = 0
        for num in count_ewt:
            f.write(f"{id}, English-EWT, {num}\n")
            id += 1
        for num in count_rand:
            f.write(f"{id}, Rand-English-EWT-CF, {num}\n")
            id += 1
        for num in count_uniform:
            f.write(f"{id}, Rand-Uniform, {num}\n")
            id += 1
        for num in count_markov2:
            f.write(f"{id}, Rand-Markov2, {num}\n")
            id += 1
        for num in count_markov3:
            f.write(f"{id}, Rand-Markov3, {num}\n")
            id += 1
        for num in count_markov5:
            f.write(f"{id}, Rand-Markov5, {num}\n")
            id += 1
        for num in count_markov10:
            f.write(f"{id}, Rand-Markov10, {num}\n")
            id += 1
        for num in count_chatgpt:
            f.write(f"{id}, Rand-ChatGPT, {num}\n")
            id += 1


    tree_ewt = [conll_to_zss_unlabel(conll.to_tree()) for conll in ewt]
    tree_rand = [conll_to_zss_unlabel(conll.to_tree()) for conll in rand]
    tree_uniform = [conll_to_zss_unlabel(conll.to_tree()) for conll in uniform]
    tree_markov2 = [conll_to_zss_unlabel(conll.to_tree()) for conll in markov2]
    tree_markov3 = [conll_to_zss_unlabel(conll.to_tree()) for conll in markov3]
    tree_markov5 = [conll_to_zss_unlabel(conll.to_tree()) for conll in markov5]
    tree_markov10 = [conll_to_zss_unlabel(conll.to_tree()) for conll in markov10]
    tree_chatgpt = [conll_to_zss_unlabel(conll.to_tree()) for conll in chatgpt]



    distArray = calcDist(tree_ewt + tree_rand + tree_uniform +
                        tree_markov2 + tree_markov3 + tree_markov5 +
                        tree_markov10 + tree_chatgpt)
    np.save(wasserTest_defs.DISTMX_DIR+"randsAll_over5.npy", distArray)

    return


def gen_distmx_rands():

    #N = 20
    K = 200

    ewt = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-EWT"])
    rand = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-English-EWT-CF"])
    chatgpt = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-ChatGPT"])



    ewt = [conll for conll in ewt if len(conll) > 5]  # Filter out too short sentences
    rand = [conll for conll in rand if len(conll) > 5]
    chatgpt = [conll for conll in chatgpt if len(conll) > 5]


    ewt = random.sample(ewt, k=K)
    count_ewt = [len(conll) for conll in ewt]

    rand = random.sample(rand, k=K)
    count_rand = [len(conll) for conll in rand]

    chatgpt = random.sample(chatgpt, k=K)
    count_chatgpt = [len(conll) for conll in chatgpt]


    with open(wasserTest_defs.LOG_DIR+"rands_counts_over5.csv", "w") as f:
        f.write("id, type, tree size\n")
        id = 0
        for num in count_ewt:
            f.write(f"{id}, English-EWT, {num}\n")
            id += 1
        for num in count_rand:
            f.write(f"{id}, Rand-English-EWT-CF, {num}\n")
            id += 1
        for num in count_chatgpt:
            f.write(f"{id}, Rand-ChatGPT, {num}\n")
            id += 1


    tree_ewt = [conll_to_zss_unlabel(conll.to_tree()) for conll in ewt]
    tree_rand = [conll_to_zss_unlabel(conll.to_tree()) for conll in rand]
    tree_chatgpt = [conll_to_zss_unlabel(conll.to_tree()) for conll in chatgpt]



    distArray = calcDist(tree_ewt + tree_rand + tree_chatgpt)
    np.save(wasserTest_defs.DISTMX_DIR+"rands_over5.npy", distArray)

    return



def gen_distmx_parser():

    #N = 20
    K = 200

    ewt = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-EWT"])
    udpipe = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-EWT-udpipe"])
    spacy = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["English-EWT-spacy"])
    chatgpt = pyconll.load_from_file(wasserTest_defs.CORPORA_PATH["Rand-ChatGPT"])



    ewt = [conll for conll in ewt if len(conll) > 5]  # Filter out too short sentences
    udpipe = [conll for conll in udpipe if len(conll) > 5]
    spacy = [conll for conll in spacy if len(conll) > 5]
    chatgpt = [conll for conll in chatgpt if len(conll) > 5]


    ewt = random.sample(ewt, k=K)
    count_ewt = [len(conll) for conll in ewt]
    udpipe = random.sample(udpipe, k=K)
    count_udpipe = [len(conll) for conll in udpipe]
    spacy = random.sample(spacy, k=K)
    count_spacy = [len(conll) for conll in spacy]
    chatgpt = random.sample(chatgpt, k=K)
    count_chatgpt = [len(conll) for conll in chatgpt]


    with open(wasserTest_defs.LOG_DIR+"parser_counts_over5.csv", "w") as f:
        f.write("id, type, tree size\n")
        id = 0
        for num in count_ewt:
            f.write(f"{id}, English-EWT, {num}\n")
            id += 1
        for num in count_udpipe:
            f.write(f"{id}, English-EWT-udpipe, {num}\n")
            id += 1
        for num in count_spacy:
            f.write(f"{id}, English-EWT-spacy, {num}\n")
            id += 1
        for num in count_chatgpt:
            f.write(f"{id}, Rand-ChatGPT, {num}\n")
            id += 1


    tree_ewt = [conll_to_zss_unlabel(conll.to_tree()) for conll in ewt]
    tree_udpipe = [conll_to_zss_unlabel(conll.to_tree()) for conll in udpipe]
    tree_spacy = [conll_to_zss_unlabel(conll.to_tree()) for conll in spacy]
    tree_chatgpt = [conll_to_zss_unlabel(conll.to_tree()) for conll in chatgpt]



    distArray = calcDist(tree_ewt + tree_udpipe + tree_spacy + tree_chatgpt)
    np.save(wasserTest_defs.DISTMX_DIR+"parser_over5.npy", distArray)

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
    ax.scatter(X_reduced[470:670, 0], X_reduced[470:670, 1], c='orange', label='Rand-English-EWT-CF')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
    plt.savefig(output_path)

    print(f"Saved to {output_path}.")
    return




if __name__ == "__main__":
    #gen_distmx_english()
    gen_distmx_multiple()
    gen_distmx_parser()
    gen_distmx_randsAll()
    gen_distmx_rands()
    #plot_multiple(wasserTest_defs.DISTMX_DIR+"distmx_multiple.npy", output_path="multiple_tsne.png")
    #plot_english(wasserTest_defs.DISTMX_DIR+"distmx_english.npy", output_path="english_tsne.png")

