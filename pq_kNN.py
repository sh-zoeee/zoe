from scripts import w_pq, dist, kNN, trees
import torch

from tqdm import tqdm

import numpy as np
from collections import Counter

import pyconll
from pqgrams.PQGram import Profile


def pqgram_distance_batch(batch1: torch.Tensor, batch2: torch.Tensor):
    """
    batch1: [batch_size, dim]
    batch2: [batch_size, dim]
    """
    assert batch1.device == batch2.device
    assert batch1.shape == batch2.shape

    # 最小値をバッチ全体で計算
    tensor_min = torch.minimum(batch1, batch2)

    # 差分を計算
    tensor_diff = (batch1 + batch2 - 2 * tensor_min)

    # 各バッチに対して内積を計算 (バッチ全体での距離)
    distances = torch.sum(tensor_diff, dim=1)  # バッチの次元で距離を計算

    return distances


def minimum_pq_dist_id(tensor_key:torch.Tensor, tensor_list:list):

    tensor_key = tensor_key
    minimum_id = 0
    minimum_distance = dist.pqgram_distance_tensor(tensor_key, tensor_list[0])
    for i in range(1, len(tensor_list)):
        distance = dist.pqgram_distance_tensor(tensor_key, tensor_list[i])
        if distance < minimum_distance:
            minimum_distance = distance
            minimum_id = i
    
    return minimum_id



def main():

    #CORPORA =   ["corpora/English-EWT.conllu", "corpora/English-EWT.conllu"]
    #LABELS = ["EWT", "EWT2"]

    """
        En_corpora.pyで生成したtensorなどのデータをそのまま用いる
    """

    train_tensors_path = "data/train_tensors_en_corpora_EWT_EWT_unlabel_10.pt"
    valid_tensors_path = "data/valid_tensors_en_corpora_EWT_EWT_unlabel_10.pt"
    test_tensors_path = "data/test_tensors_en_corpora_EWT_EWT_unlabel_10.pt"

    train_tensors = torch.load(train_tensors_path) + torch.load(valid_tensors_path)
    test_tensors = torch.load(test_tensors_path)
    

    train_labels_path = "data/train_labels_en_corpora_EWT_EWT_unlabel_10.pt"
    valid_labels_path = "data/valid_labels_en_corpora_EWT_EWT_unlabel_10.pt"
    test_labels_path = "data/test_labels_en_corpora_EWT_EWT_unlabel_10.pt"
    
    #train_indexes = torch.load(train_indexes_path).tolist() + torch.load(valid_indexes_path).tolist()
    #test_indexes = torch.load(test_indexes_path).tolist()
    train_labels = torch.load(train_labels_path) + torch.load(valid_labels_path)
    test_labels = torch.load(test_labels_path)

    test_size = len(test_labels)
    error = 0

    for i in tqdm(range(test_size)):
        pred_id = minimum_pq_dist_id(test_tensors[i], train_tensors)
        pred_label = train_labels[pred_id]
        if pred_label != test_labels[i]:
            error += 1

    print(f"error: {error}")
    print(f"error rate: {error/test_size:.2f}")

    #CORPUS_LIST = []
    #for corpus in CORPORA:
    #    CORPUS_LIST.append(corpus.split(".")[0].split("/")[-1])
    """
    CORPUS_i_LENGTH = []

    CoNLL = []
    labels = []
    for i in range(len(CORPORA)):
        tmp_conll = pyconll.load_from_file(CORPORA[i])
        CoNLL += tmp_conll
        for _ in range(len(tmp_conll)):
            labels.append(LABELS[i])
        if i != 0:
            CORPUS_i_LENGTH.append(len(tmp_conll)+CORPUS_i_LENGTH[i-1])
        else:
            CORPUS_i_LENGTH.append(len(tmp_conll))
    
    print(CORPUS_i_LENGTH)
    
    num_trees = CORPUS_i_LENGTH[-1]
    pqtrees = [trees.conllTree_to_pqTree_upos(conll.to_tree()) for conll in CoNLL]
    #pqtrees = [trees.conllTree_to_pqTree_unlabeled(conll.to_tree()) for conll in CoNLL]
    
    pqIndex = [Profile(tree, p=2, q=2) for tree in pqtrees]

    train_pqIndexes = []

    for train_id in train_indexes:
        train_pqIndexes.append(pqIndex[int(train_id)])
    
    error = 0
    test_size = 2000#len(test_indexes)
    print(f'test size: {test_size}')
    for i, test_id in enumerate(tqdm(test_indexes[:1000])):
        test_id = int(test_id)

        test_pqIndex = pqIndex[test_id]
        pred_label = kNN.kNN_pqgram_set(test_pqIndex, train_pqIndexes, train_labels, k=1)

        if pred_label != labels[test_id]:
            error += 1
    
    print(f'error: {error}')
    print(f'error rate: {error/test_size}')
    """ 



if __name__=="__main__":
    main()