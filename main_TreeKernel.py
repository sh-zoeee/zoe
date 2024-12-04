import numpy as np
from tqdm import tqdm, trange
from time import time
from collections import Counter
from time import time

import pyconll
from scripts.TreeKernel import tree, tree_kernels

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def to_prolog_unlabel(tree: pyconll.tree.tree.Tree) -> str:

    if tree._children:
        children_repr = ', '.join(to_prolog_unlabel(child) for child in tree._children)
        return f'_({children_repr})'
    else:
        return f'_'
    

def to_prolog_upos(tree: pyconll.tree.tree.Tree) -> str:

    if tree._children:
        children_repr = ', '.join(to_prolog_upos(child) for child in tree._children)
        return f'{tree.data.upos}({children_repr})'
    else:
        return f'{tree.data.upos}'


def calc_kernel_matrix(data, kernel: tree_kernels.Kernel):
    n = len(data)
    matrix = np.zeros((n,n))
    start = time()
    for i in range(n):
        if i%2000 == 0:
            print(f"{i}th loop, {time()-start}sec")
        for j in range(i,n):
            matrix[i][j] = kernel.kernel(data[i], data[j])
    return matrix + matrix.T -np.diag(matrix.diagonal())


def calc_kernel_matrix(kernel: tree_kernels.Kernel, data1, data2=None):
    
    if data2==None:
        n = len(data1)
        matrix = np.zeros((n,n))
        start = time()
        for i in trange(n, desc="[matrix for train]"):
            if i%2000 == 0:
                print(f"{i}th loop, {time()-start}sec", flush=True)
            for j in range(i,n):
                matrix[i][j] = kernel.kernel(data1[i], data1[j])
        return matrix + matrix.T -np.diag(matrix.diagonal())

    n1, n2 = len(data1), len(data2)
    matrix = np.zeros((n1, n2))
    start = time()
    for i in trange(n1, desc="[matrix for test]"):
        if i%2000 == 0:
            print(f"{i}th loop, {time()-start}sec", flush=True)
        for j in range(n2):
            matrix[i][j] = kernel.kernel(data1[i], data2[j])
    return matrix



def main():

    random_state = 10
    
    start = time()

    CoNLL = pyconll.load_from_file("corpora/English/English-EWT.conllu")
    count_en = len(CoNLL)
    CoNLL += pyconll.load_from_file("corpora/English/English-Atis.conllu")
    count_ja = len(CoNLL) - count_en

    if count_en==count_ja:
        labels = [0]*count_en + [1]*count_ja
    elif count_en < count_ja:
        CoNLL = CoNLL[:2*count_en]
        labels = [0]*count_en + [1]*count_en
    else:
        CoNLL = CoNLL[count_en-count_ja:]
        labels = [0]*count_ja + [1]*count_ja


    

    trees = [conll.to_tree() for conll in CoNLL]

    data = []

    for t in trees:
        root = tree.TreeNode.fromPrologString(to_prolog_upos(t))
        tree_ = tree.Tree(root)
        data.append(tree_)


    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.20, random_state=random_state
    )

    # 部分木の比較用カーネルを作成
    lambda_value = 0.5
    mu_value = 0.5

    print(f"lambda=(mu)={lambda_value}")
    kernel = tree_kernels.KernelST(lambda_value, mu_value)

    train_kernel_matrix = calc_kernel_matrix(kernel, train_data)

    test_kernel_matrix = calc_kernel_matrix(kernel, test_data, train_data)


    model = SVC(kernel='precomputed')
    model.fit(train_kernel_matrix, train_labels)

    test_pred = model.predict(test_kernel_matrix)

    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_pred))



    print(classification_report(test_labels, test_pred))

    acc = accuracy_score(test_labels, test_pred)

    print(f"acc: {acc}")
    print(f"error rate: {1-acc}")


    end = time()

    print(f"time: {end-start}sec")

    return

if __name__ == '__main__':
    main()