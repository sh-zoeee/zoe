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

    TREE_SIZE = 20

    SAMPLING_SIZE = 10000
    MCMC_SIZE = 100
    NL_SIZE = 100

    random.seed(0)
    
    trees_en = random.sample(pyconll.load_from_file(EN_PATH), k=NL_SIZE)
    trees_en = [trees.conllTree_to_zssNode_unlabel(t.to_tree()) for t in trees_en]
    # diameter_en = calc_diameter(trees_en)
    # print("English-EWT,\t", diameter_en)
    """自然言語の直径の計算
    trees_ja = random.sample(pyconll.load_from_file(JA_PATH), k=NL_SIZE)
    trees_ja = [trees.conllTree_to_zssNode_unlabel(t.to_tree()) for t in trees_ja]
    diameter_ja = calc_diameter(trees_ja)
    print("Japanese-GSDLUW,\t", diameter_ja)
    
    trees_fr = random.sample(pyconll.load_from_file(FR_PATH), k=NL_SIZE)
    trees_fr = [trees.conllTree_to_zssNode_unlabel(t.to_tree()) for t in trees_fr]
    diameter_fr = calc_diameter(trees_fr)
    print("French-GSD,\t", diameter_fr)

    trees_ch = random.sample(pyconll.load_from_file(CH_PATH), k=NL_SIZE)
    trees_ch = [trees.conllTree_to_zssNode_unlabel(t.to_tree()) for t in trees_ch]
    diameter_ch = calc_diameter(trees_ch)
    print("Chinese-GSD,\t", diameter_ch)
    """

    """MCMCの直径の計算"""
    # 初期状態が完全二分木のMCMC
    tree = mcmc.generate_full_binary_tree(TREE_SIZE)
    trees_mcmc_b = []

    for step in range(SAMPLING_SIZE):
        tree = mcmc.propose_subtree_move(tree)
        trees_mcmc_b.append(tree) # このtreeは次のstepでも使う 

    random.seed(0)
    #trees_mcmc_b = random.sample(trees_mcmc_b, k=MCMC_SIZE)
    trees_mcmc_b = trees_mcmc_b[-MCMC_SIZE:]
    trees_mcmc_b = [mcmc.dict_to_zss(t) for t in trees_mcmc_b]

    diameter_mcmc_b = calc_diameter(trees_mcmc_b)
    print(f"MCMC_bi({SAMPLING_SIZE}),\t{diameter_mcmc_b}")

    # 初期状態がEWTの木のMCMC
    tree = mcmc.generate_full_binary_tree(TREE_SIZE)
    trees_mcmc_nl = []

    for step in range(SAMPLING_SIZE):
        tree = mcmc.propose_subtree_move(tree)
        trees_mcmc_nl.append(tree) # このtreeは次のstepでも使う 

    random.seed(0)
    #trees_mcmc_nl = random.sample(trees_mcmc_nl, k=MCMC_SIZE)
    trees_mcmc_nl = trees_mcmc_nl[-MCMC_SIZE:]
    trees_mcmc_nl = [mcmc.dict_to_zss(t) for t in trees_mcmc_nl]

    diameter_mcmc_nl = calc_diameter(trees_mcmc_nl)
    print(f"MCMC_nl({SAMPLING_SIZE}),\t{diameter_mcmc_nl}")

    # 初期状態がスター木のMCMC
    tree = mcmc.generate_star_tree(TREE_SIZE)
    trees_mcmc_s = []

    for step in range(SAMPLING_SIZE):
        tree = mcmc.propose_subtree_move(tree)
        trees_mcmc_s.append(tree)
    
    random.seed(0)
    #trees_mcmc_s = random.sample(trees_mcmc_s, k=MCMC_SIZE)
    trees_mcmc_s = trees_mcmc_s[-MCMC_SIZE:]
    trees_mcmc_s = [mcmc.dict_to_zss(t) for t in trees_mcmc_s]
    diameter_mcmc_s = calc_diameter(trees_mcmc_s)
    print(f"MCMC_star({SAMPLING_SIZE}),\t{diameter_mcmc_s}")

    return


def calc_diameter(tree_list: list):
    """
        TEDを利用して、集合の距離を計算
    """
    tree_count = len(tree_list)
    max_distance = 0
    for i in tqdm(range(tree_count)):
        for j in range(i+1, tree_count):
            dist_ij = zss.simple_distance(tree_list[i], tree_list[j])
            if max_distance < dist_ij:
                max_distance = dist_ij
    return max_distance


if __name__ == "__main__":
    main()
