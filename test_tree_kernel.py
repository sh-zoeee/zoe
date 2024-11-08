import numpy as np
from tqdm import tqdm
from time import time

import pyconll
from scripts.TreeKernel import tree, tree_kernels

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def to_prolog(tree: pyconll.tree.tree.Tree) -> str:

    if tree._children:
        children_repr = ', '.join(to_prolog(child) for child in tree._children)
        return f'_({children_repr})'
    else:
        return f'_'



def calc_kernel_matrix(data1, data2, kernel: tree_kernels.KernelST):
    n1, n2 = len(data1), len(data2)
    matrix = np.zeros((n1, n2))
    for i in tqdm(range(n1)):
        for j in range(n2):
            matrix[i][j] = kernel.kernel(data1[i], data2[j])
    return matrix



def main():

    random_state = 50
    
    start = time()

    CoNLL = pyconll.load_from_file("corpora/English/English-EWT.conllu")
    count_en = len(CoNLL)
    CoNLL += pyconll.load_from_file("corpora/English/English-Atis.conllu")
    count_ja = len(CoNLL) - count_en

    labels = [0]*count_en + [1]*count_ja

    trees = [conll.to_tree() for conll in CoNLL]

    data = []

    for t in trees:
        root = tree.TreeNode.fromPrologString(to_prolog(t))
        tree_ = tree.Tree(root)
        data.append(tree_)


    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.20, random_state=random_state
    )

    # 部分木の比較用カーネルを作成
    lambda_value = 0.5
    kernel = tree_kernels.KernelST(lambda_value)

    train_kernel_matrix = calc_kernel_matrix(train_data, train_data, kernel)

    test_kernel_matrix = calc_kernel_matrix(test_data, train_data, kernel)


    model = SVC(kernel='precomputed')
    model.fit(train_kernel_matrix, train_labels)

    test_pred = model.predict(test_kernel_matrix)

    print("test labels:")
    print(test_labels)

    print("prediction:")
    print(test_pred)

    print(classification_report(test_labels, test_pred))

    print(accuracy_score(test_labels, test_pred))

    end = time()

    print(f"time: {end-start}sec")

    return

if __name__ == '__main__':
    main()