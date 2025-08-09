from function import mcmc, visualize, stats_tree, trees

import numpy as np
import matplotlib.pyplot as plt

import zss, pyconll
from sklearn.manifold import TSNE

import random
from tqdm import tqdm

EN_PATH = "/home/yamazoe/zoe/data/arranged/corpora_20/English-EWT_20.conllu"
JA_PATH = "/home/yamazoe/zoe/data/arranged/corpora_20/Japanese-GSDLUW_20.conllu"
FR_PATH = "/home/yamazoe/zoe/data/arranged/corpora_20/French-GSD_20.conllu"
CH_PATH = "/home/yamazoe/zoe/data/arranged/corpora_20/Chinese-GSD_20.conllu"


def main():

    SAMPLING_SIZE = 200
    NL_SIZE = 20

    trees_en = random.sample(pyconll.load_from_file(EN_PATH), k=NL_SIZE)
    trees_en = [trees.conllTree_to_zssNode_unlabel(t.to_tree()) for t in trees_en]

    trees_ja = random.sample(pyconll.load_from_file(JA_PATH), k=NL_SIZE)
    trees_ja = [trees.conllTree_to_zssNode_unlabel(t.to_tree()) for t in trees_ja]
    
    trees_fr = random.sample(pyconll.load_from_file(FR_PATH), k=NL_SIZE)
    trees_fr = [trees.conllTree_to_zssNode_unlabel(t.to_tree()) for t in trees_fr]

    trees_ch = random.sample(pyconll.load_from_file(CH_PATH), k=NL_SIZE)
    trees_ch = [trees.conllTree_to_zssNode_unlabel(t.to_tree()) for t in trees_ch]

    

    tree = trees_en[0]
    trees_mcmc = []

    for step in range(10000):
        tree = mcmc.propose_subtree_move(tree)
        trees_mcmc.append(tree)
    
    random.seed(0)

    trees_mcmc = random.sample(trees_mcmc, k=SAMPLING_SIZE)
    trees_mcmc = [mcmc.dict_to_zss(t) for t in trees_mcmc]


    


    trees_all = trees_mcmc + trees_en + trees_ja + trees_fr + trees_ch

    len_tree = len(trees_all)
    distance_matrix = np.zeros((len_tree, len_tree))

    for i  in tqdm(range(len_tree)):
        for j in range(i+1, len_tree):
            distance_matrix[i][j] = zss.simple_distance(trees_all[i], trees_all[j])

    distance_matrix += distance_matrix.T-np.diag(distance_matrix.diagonal())

    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(distance_matrix)


    fig, ax = plt.subplots()

    ax.scatter(X_reduced[:200,0], X_reduced[:200,1], label="MCMC")
    ax.scatter(X_reduced[200:220,0], X_reduced[200:220,1], label="English")
    ax.scatter(X_reduced[220:240,0], X_reduced[220:240,1], label="Japanese")
    ax.scatter(X_reduced[240:260,0], X_reduced[240:260,1], label="French")
    ax.scatter(X_reduced[260:280,0], X_reduced[260:280,1], label="Chinese")

    plt.legend()

    plt.savefig("mcmc_binary.png")

    return


if __name__ == "__main__":
    main()
