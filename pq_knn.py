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
        En_corpora.pyで生成したtensorなどのデータをそのまま用いる
    """

    train_tensors_path = "data/train_tensors_en_corpora_En_EWT_spacy_unlabel_50.pt"
    valid_tensors_path = "data/valid_tensors_en_corpora_En_EWT_spacy_unlabel_50.pt"
    test_tensors_path = "data/test_tensors_en_corpora_En_EWT_spacy_unlabel_50.pt"

    train_tensors = torch.load(train_tensors_path) + torch.load(valid_tensors_path)
    test_tensors = torch.load(test_tensors_path)
    
    #train_indexes_path = "data/train_indexes_en_corpora_EWT_rand_unlabel_50.pt"
    #valid_indexes_path = "data/valid_indexes_en_corpora_EWT_rand_unlabel_50.pt"
    #test_indexes_path = "data/test_indexes_en_corpora_EWT_rand_unlabel_50.pt"

    train_labels_path = "data/train_labels_en_corpora_En_EWT_spacy_unlabel_50.pt"
    valid_labels_path = "data/valid_labels_en_corpora_En_EWT_spacy_unlabel_50.pt"
    test_labels_path = "data/test_labels_en_corpora_En_EWT_spacy_unlabel_50.pt"
    
    #train_indexes = torch.load(train_indexes_path).tolist() + torch.load(valid_indexes_path).tolist()
    #test_indexes = torch.load(test_indexes_path).tolist()
    train_labels = torch.load(train_labels_path) + torch.load(valid_labels_path)
    test_labels = torch.load(test_labels_path)

    LABELS = list(set(test_labels))
    assert len(LABELS) == 2

    M = [[0,0],[0,0]] # 混同行列


    test_size = len(test_labels)
    error = 0

    train_tensors = torch.stack([t.to("cuda") for t in train_tensors])
    test_tensors = torch.stack([t.to("cuda") for t in test_tensors])

    for i in tqdm(range(test_size)):

        test_tensor = test_tensors[i].unsqueeze(0)  # [1, dim]の形状に変換

        # 全ての訓練テンソルとの距離をバッチで計算
        distances = pqgram_distance_batch(train_tensors, test_tensor.repeat(train_tensors.size(0), 1))

        # 最小の距離を持つテンソルを見つける
        #pred_id = torch.argmin(distances).item()
        _, pred_id = torch.kthvalue(distances, k=2)
        pred_label = train_labels[pred_id]

        # 予測が間違っていたらエラー数を増加
        if pred_label == LABELS[0]:
            if test_labels[i] == LABELS[0]:
                M[0][0] += 1
            elif test_labels[i] == LABELS[1]:
                M[0][1] += 1
                error += 1
        elif pred_label == LABELS[1]:
            if test_labels[i] == LABELS[0]:
                M[1][0] += 1
                error += 1
            elif test_labels[i] == LABELS[1]:
                M[1][1] += 1

    print(M)
    print(f"error: {error}")
    print(f"error rate: {error/test_size:.2f}")
    TP = M[0][0]/(M[0][0]+M[0][1])
    FP = M[0][0]/(M[0][0]+M[1][0])

    print(f'f1 score: {2*TP*FP/(TP+FP):.3f}\n')

    



if __name__=="__main__":
    main()