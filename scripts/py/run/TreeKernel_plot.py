from tqdm import tqdm, trange
from time import time
import numpy as np

import pyconll
from scripts.TreeKernel import tree, tree_kernels, utility

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import japanize_matplotlib

import random



def main():

    nums_list = []

    random.seed(10)
    k = 1000

    CoNLL_multi = pyconll.load_from_file("corpora/English/English-EWT.conllu")
    CoNLL_multi = random.sample(CoNLL_multi, k=k)
    num_En = len(CoNLL_multi)
    nums_list.append(num_En)

    CoNLL_Ja = pyconll.load_from_file("corpora/Japanese/Japanese-BCCWJ.conllu")
    CoNLL_multi += random.sample(CoNLL_Ja, k=k)
    num_Ja = len(CoNLL_multi) - sum(nums_list)
    nums_list.append(num_Ja)

    CoNLL_Fr = pyconll.load_from_file("corpora/French/French-GSD.conllu")
    CoNLL_multi += random.sample(CoNLL_Fr, k=k)
    num_Fr = len(CoNLL_multi) - sum(nums_list)
    nums_list.append(num_Fr)

    CoNLL_Zh = pyconll.load_from_file("corpora/Chinese/Chinese-GSD.conllu")
    CoNLL_multi += random.sample(CoNLL_Zh, k=k)
    num_Zh = len(CoNLL_multi) - sum(nums_list)
    nums_list.append(num_Zh)

    CoNLL_Ko = pyconll.load_from_file("corpora/Korean/Korean-Kaist.conllu")
    CoNLL_multi += random.sample(CoNLL_Ko, k=k)
    num_Ko = len(CoNLL_multi) - sum(nums_list)
    nums_list.append(num_Ko)

    print(f"Total Trees: {sum(nums_list)}")
    print(f"{nums_list}")


    labels = []
    for i, nums in enumerate(nums_list):
        labels += [i]*nums


    trees = [conll.to_tree() for conll in CoNLL_multi]

    data = []

    for t in trees:
        root = tree.TreeNode.fromPrologString(utility.to_prolog_upos(t))
        tree_ = tree.Tree(root)
        data.append(tree_)

    lambda_value = 0.5
    kernel = tree_kernels.KernelST(lambda_value)

    kernel_matrix_multi = utility.calc_kernel_matrix(kernel=kernel, data1=data)
    np.save(f"kernel_matrix_multi_languages_{k}_zh_upos", kernel_matrix_multi)

    return 


if __name__ == "__main__":
    main()