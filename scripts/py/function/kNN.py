from . import dist

import torch
from collections import Counter


def kNN_pqgram(test_tensor, train_tensors, train_labels, k):

    # 各訓練データとの距離を計算
    distances = []
    for tensor, label in zip(train_tensors, train_labels):
        distance = dist.pqgram_distance_tensor(test_tensor, tensor)
        distances.append(distance)
    
    indexes = torch.topk(torch.tensor(distances), k, largest=False)[1]
    classes = [train_labels[i] for i in indexes]

    return Counter(classes).most_common(1)[0][0]


def kNN_pqgram_set(test_pqIndex, train_pqIndexes, train_labels, k):

    distances = []
    for train_pqIndex in train_pqIndexes:
        distance = dist.pqgram_distance(test_pqIndex, train_pqIndex)
        distances.append(distance)

    indexes = torch.topk(torch.tensor(distances), k, largest=False)[1]
    classes = [train_labels[i] for i in indexes]

    return Counter(classes).most_common(1)[0][0]
