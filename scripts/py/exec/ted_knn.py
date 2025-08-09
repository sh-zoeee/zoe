from scripts import trees
import torch

from tqdm import tqdm

import numpy as np
from statistics import mode
from collections import Counter
from sklearn.model_selection import train_test_split

import pyconll
from pqgrams.PQGram import Profile

import zss



def main():

    CORPORA =   ["corpora/English-EWT.conllu", "parsed/En-EWT-spacy.conllu"]
    LABELS = ["EWT", "spacy"]

    CoNLLU = pyconll.load_from_file(CORPORA[0])
    EWT_COUNT = len(CoNLLU)
    CoNLLU += pyconll.load_from_file(CORPORA[1])
    another_COUNT = len(CoNLLU)-EWT_COUNT

    labels = [LABELS[0]]*EWT_COUNT + [LABELS[1]]*another_COUNT

    indexes = list(range(len(CoNLLU)))


    id_train, id_test, labels_train, labels_test = train_test_split(indexes, labels, test_size=0.4, random_state=50) # 無印はrandom state = 42
    id_val, id_test, labels_val, labels_test = train_test_split(id_test, labels_test, test_size=0.5, random_state=50)

    id_train, _, labels_train, _ = train_test_split(id_train, labels_train, train_size=3000, random_state=50)


    trees_train = []
    for id in tqdm(id_train, desc="train trees"):
        trees_train.append(trees.conllTree_to_zssNode_unlabel(CoNLLU[id].to_tree()))

    trees_test = []
    for id in tqdm(id_test, desc="test trees"):
        trees_test.append(trees.conllTree_to_zssNode_unlabel(CoNLLU[id].to_tree()))
        

    #t1 = trees.conllTree_to_zssNode_unlabel(CoNLLU[0].to_tree())
    #t2 = trees.conllTree_to_zssNode_unlabel(CoNLLU[1].to_tree())

    #print(zss.simple_distance(t1,t2))


    M = [[0,0],[0,0]] # 混同行列

    error = 0

    test_size = len(labels_test)

    for i in tqdm(range(100)):

        test_tree = trees_test[i]
        distances = []


        # 全ての訓練テンソルとの距離をバッチで計算
        for j in tqdm(range(len(id_train)), desc=f"{i}th loop", leave=False):
            dist = int(zss.simple_distance(trees_train[j], test_tree))
            distances.append(dist)
        
        id_1st = distances.index(min(distances))
        distances.pop(id_1st)
        id_2nd = distances.index(min(distances))
        distances.pop(id_2nd)
        id_3rd = distances.index(min(distances))
        distances.pop(id_3rd)

        pred_label = mode([labels_train[id_1st], labels_train[id_2nd], labels_train[id_3rd]])

        # 予測が間違っていたらエラー数を増加
        if pred_label == LABELS[0]:
            if labels_test[i] == LABELS[0]:
                M[0][0] += 1
            elif labels_test[i] == LABELS[1]:
                M[0][1] += 1
                error += 1
        elif pred_label == LABELS[1]:
            if labels_test[i] == LABELS[0]:
                M[1][0] += 1
                error += 1
            elif labels_test[i] == LABELS[1]:
                M[1][1] += 1

    print(M)
    print(f"error: {error}")
    print(f"error rate: {error/test_size:.2f}")
    TP = M[0][0]/(M[0][0]+M[0][1])
    FP = M[0][0]/(M[0][0]+M[1][0])

    print(f'f1 score: {2*TP*FP/(TP+FP):.3f}\n')

    



if __name__=="__main__":
    main()