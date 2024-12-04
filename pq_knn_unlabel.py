from scripts import w_pq, dist, kNN, trees
import torch

from tqdm import tqdm
from time import time

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
    tensor_diff = batch1 + batch2 - 2 * tensor_min

    # 各バッチに対して内積を計算 (バッチ全体での距離)
    distances = torch.sum(tensor_diff, dim=1)  # バッチの次元で距離を計算

    return distances


def pqgram_distance_tensor(tensor1: torch.Tensor, tensor2: torch.Tensor):

    assert tensor1.device == tensor2.device

    device = tensor1.device
    
    dim = tensor1.size()[0]

    tensor_min = torch.minimum(tensor1, tensor2)
    tensor_diff = (tensor1 + tensor2 - 2*tensor_min)

    d = torch.dot(torch.ones(dim, dtype=torch.float32, device=device), tensor_diff)
    return d


def minimum_pq_dist_id(tensor_key:torch.Tensor, tensor_list:list):

    minimum_id = 0
    minimum_distance = pqgram_distance_tensor(tensor_key, tensor_list[0])
    for i in range(1, len(tensor_list)):
        distance = pqgram_distance_tensor(tensor_key, tensor_list[i])
        if distance < minimum_distance:
            minimum_distance = distance
            minimum_id = i
    
    return minimum_id



def main():

    #CORPORA =   ["corpora/English-EWT.conllu", "corpora/English-EWT.conllu"]
    #LABELS = ["EWT", "EWT2"]

    """
        en_corpora.pyで生成したtensorなどのデータをそのまま用いる
    """

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--source","-s",help="比較元")
    argparser.add_argument("--target","-t",help="比較対象")
    args = argparser.parse_args()

    source = args.source
    target = args.target

    random_state = 50
    label_type = "upos"


    train_tensors_path = f"data/train_tensors_en_corpora_{source}_{target}_{label_type}_{str(random_state)}.pt"
    valid_tensors_path = f"data/valid_tensors_en_corpora_{source}_{target}_{label_type}_{str(random_state)}.pt"
    test_tensors_path = f"data/test_tensors_en_corpora_{source}_{target}_{label_type}_{str(random_state)}.pt"

    train_tensors = torch.load(train_tensors_path) + torch.load(valid_tensors_path)
    test_tensors = torch.load(test_tensors_path)

    train_labels_path = f"data/train_labels_en_corpora_{source}_{target}_{label_type}_{str(random_state)}.pt"
    valid_labels_path = f"data/valid_labels_en_corpora_{source}_{target}_{label_type}_{str(random_state)}.pt"
    test_labels_path = f"data/test_labels_en_corpora_{source}_{target}_{label_type}_{str(random_state)}.pt"

    train_labels = torch.load(train_labels_path) + torch.load(valid_labels_path)
    test_labels = torch.load(test_labels_path)

    LABELS = list(set(test_labels))

    test_size = len(test_labels)

    M = [[0,0],[0,0]]

    train_tensors = torch.stack([t.to("cuda") for t in train_tensors])
    test_tensors = torch.stack([t.to("cuda") for t in test_tensors])

    for i in tqdm(range(test_size)):

        test_tensor = test_tensors[i].unsqueeze(0)  # [1, dim]の形状に変換

        # 全ての訓練テンソルとの距離をバッチで計算
        distances = pqgram_distance_batch(
            train_tensors, 
            test_tensor.repeat(train_tensors.size(0), 1)
        )

        # 最小の距離を持つテンソルを見つける
        pred_id = torch.argmin(distances).item()

        if distances[pred_id]==0:
            zero_dists = torch.nonzero(distances == 0, as_tuple=False).to("cpu").detach().numpy()
            zero_id = np.ndarray(len(zero_dists), dtype=int)
            for i, id in enumerate(zero_dists):
                zero_id[i] = int(id[0])
            pred_id = np.random.choice(zero_id, size=1)[0]

        pred_label = train_labels[pred_id]

        # 予測が間違っていたらエラー数を増加
        if pred_label == LABELS[0]:
            if test_labels[i] == LABELS[0]:
                M[0][0] += 1
            elif test_labels[i] == LABELS[1]:
                M[0][1] += 1
        elif pred_label == LABELS[1]:
            if test_labels[i] == LABELS[0]:
                M[1][0] += 1
            elif test_labels[i] == LABELS[1]:
                M[1][1] += 1
        
    error = M[1][0] + M[0][1]

    print(f"混同行列M:{M}")

    print(f"error: {error}")
    print(f"error rate: {error/test_size:.2f}")

    TP = M[0][0]/(M[0][0]+M[0][1])
    FP = M[0][0]/(M[0][0]+M[1][0])

    print(f'f1 score: {2*TP*FP/(TP+FP):.3f}\n')


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
    for i, test_id in enumerate(tqdm(test_indexes[:2000])):
        test_id = int(test_id)

        test_pqIndex = pqIndex[test_id]
        pred_label = kNN.kNN_pqgram_set(test_pqIndex, train_pqIndexes, train_labels, k=1)

        if pred_label != labels[test_id]:
            error += 1
    
    print(f'error: {error}')
    print(f'error rate: {error/test_size}')
    """ 



if __name__=="__main__":
    start = time()
    main()
    end = time()
    print(f"time: {end-start}sec")