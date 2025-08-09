from function import mcmc, trees

import pyconll, zss

from tqdm import tqdm

import random
import numpy as np


PATH_ENGLISH_20 = "/home/yamazoe/zoe/data/arranged/corpora_20/English-EWT_20.conllu"
PATH_JAPANESE_20 = "/home/yamazoe/zoe/data/arranged/corpora_20/Japanese-GSDLUW_20.conllu"
PATH_CHINESE_20 = "/home/yamazoe/zoe/data/arranged/corpora_20/Chinese-GSD_20.conllu"
PATH_KOREAN_20 = "/home/yamazoe/zoe/data/arranged/corpora_20/Korean-Kaist_20.conllu"
PATH_FRENCH_20 = "/home/yamazoe/zoe/data/arranged/corpora_20/French-GSD_20.conllu"

PATH_ENGLISH_EWT = "/home/yamazoe/zoe/data/original/treebank/English/English-EWT.conllu"

DIST_MX_DIR = "/home/yamazoe/zoe/data/numpy_data/distmx/ewt/"
SCATTER_DIR = "/home/yamazoe/zoe/figures/scatter/MCMC/"

M = 100
J = 10
K = 10000

def main(n:int):
    N = n
    print(f"=== N = {N} ===", flush=True)

    conll_en = pyconll.load_from_file(PATH_ENGLISH_EWT)
    tree_en_N = []
    for conll in conll_en:
        if len(conll) == N:
            tree_en_N.append(conll.to_tree())
    random.seed(0)
    tree_en_N = random.sample(tree_en_N, k=100)
    tree_en_zss = [trees.conllTree_to_zssNode_unlabel(tree) for tree in tree_en_N]


    tree_count = len(tree_en_zss)
    distance_matrix = np.zeros((tree_count, tree_count))
    for i in tqdm(range(tree_count)):
        for j in range(i, tree_count):
            distance_matrix[i, j] = zss.simple_distance(tree_en_zss[i], tree_en_zss[j])
        #if i % 10 == 0:
        #    print(f"{i} / {tree_count}")
        
    distance_matrix = distance_matrix+distance_matrix.T-np.diag(distance_matrix.diagonal())
    np.save(DIST_MX_DIR +f"rand_{N}.npy", distance_matrix)




    return 

if __name__ == "__main__":
    #from argparse import ArgumentParser
    #parser = ArgumentParser()
    #parser.add_argument("--N", "-n", type=int, default=20, help="木のサイズ")
    #args = parser.parse_args()

    main(16)
    main(17)
    main(18)
    main(19)
    main(20)
