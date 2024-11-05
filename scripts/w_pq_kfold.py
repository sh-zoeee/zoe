
import numpy as np
from sklearn.model_selection import KFold
import pyconll
from pqgrams.PQGram import Profile
from . import pq_gram, trees, func
import torch
from torch import nn, optim
from random import choices
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from time import time


def weighted_pqgram_distance_batch(weights, batch1: torch.Tensor, batch2: torch.Tensor):
    assert batch1.device == batch2.device
    assert batch1.shape == batch2.shape

    device = batch1.device
    min12 = torch.minimum(batch1, batch2)
    diff = batch1+batch2-2*min12
    aw = func.softplus(weights.to(device))
    return (diff*aw).sum(dim=1)


# weighted pq-gram distance の計算
def weighted_pqgram_distance(weights, tensor1: torch.Tensor, tensor2: torch.Tensor):
    device = tensor1.device
    min12 = torch.minimum(tensor1, tensor2).to(device)
    diff = tensor1 + tensor2 - 2 * min12
    aw = func.softplus(weights).to(device)
    return torch.dot(diff, aw)

# weighted pq-gram distance の距離行列の作成
def distance_matrix(tensors: list, weights):
    num_tensor = len(tensors)
    dist_mat = torch.zeros(num_tensor, num_tensor, dtype=torch.float32)
    for i in range(num_tensor):
        for j in range(i + 1, num_tensor):
            dist = weighted_pqgram_distance(weights, tensors[i], tensors[j])
            dist_mat[i, j] = dist
            dist_mat[j, i] = dist
    return dist_mat



def create_pairs_lmnn(data, labels, weights, k):

    #distances = distance_matrix(data, weights)
    upper_bound = 1000
    train_tensors = torch.stack([t for t in choices(data, k=upper_bound)])

    positive_pairs = []
    negative_pairs = []

    for i in range(upper_bound):

        data_i = train_tensors[i]

        distances = weighted_pqgram_distance_batch(weights, train_tensors, data_i.repeat(train_tensors.size(0), 1))

        distances_asc_arg = torch.argsort(distances)[1:]

        targets = []
        impostors = []
        label_i = labels[i]

        j = 0
        while len(targets)<=k and j<len(data):
            idx = distances_asc_arg[j]
            if labels[idx] == label_i:
                targets.append(idx)
            else:
                impostors.append(idx)
            j += 1
    
        positive_pairs.extend([(data[i], data[j]) for j in targets])
        negative_pairs.extend([(data[i], data[j]) for j in impostors])

    del distances, distances_asc_arg
    torch.cuda.empty_cache()

    return positive_pairs, negative_pairs

"""


def create_pairs_lmnn(data, labels, weights, k):

    #distances = distance_matrix(data, weights)
    distances = distance_matrix_chunked(
        data.to("cuda" if torch.cuda.is_available() else "cpu"), 
        weights, 
        chunk_size=int(len(data)/512)
    )  

    positive_pairs = []
    negative_pairs = []

    for i in range(len(data)):

        distances_asc_arg = torch.argsort(distances[i])[1:]

        targets = []
        impostors = []
        label_i = labels[i]
        j = 0
        while len(targets)<=k and j<len(data):
            idx = distances_asc_arg[j]
            if labels[idx] == label_i:
                targets.append(idx)
            else:
                impostors.append(idx)
            j += 1
    
    positive_pairs.extend([(data[i], data[j]) for j in targets])
    negative_pairs.extend([(data[i], data[j]) for j in impostors])

    del distances, distances_asc_arg
    torch.cuda.empty_cache()

    return positive_pairs, negative_pairs
"""
# 
class WeightedPqgramDistance(nn.Module):
  
    def __init__(self, dimension, positive_pairs, negative_pairs):
        super(WeightedPqgramDistance, self).__init__()
        self.weights = nn.Parameter(torch.ones(dimension))
        self.positive_pairs = positive_pairs
        self.negative_pairs = negative_pairs
    
    def forward(self, tensor1, tensor2):
        dist = weighted_pqgram_distance(self.weights, tensor1, tensor2)
        return dist
    
    def setPairs(self, train_tensors, train_labels, k):
        self.positive_pairs, self.negative_pairs = create_pairs_lmnn(train_tensors, train_labels, self.weights, k)
  

# 損失関数の定義
class MetricLearingLoss(nn.Module):
    def __init__(self, margin1, margin2, beta):
        super(MetricLearingLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.beta = beta
    
    def forward(self, dist_func, positive_pairs, negative_pairs):
        loss = 0.0

        for (tensor1, tensor2) in positive_pairs:
            dist = dist_func(tensor1, tensor2)
            if dist > self.margin1:
                loss += dist - self.margin1
    
        for (tensor1, tensor2) in negative_pairs:
            dist = dist_func(tensor1, tensor2)
            if dist < self.margin2:
                loss += self.margin2 - dist
    
        reg_term = torch.norm(dist_func.weights)**2

        loss += reg_term.to("cpu")* self.beta

        return loss


# 5-fold cross-validation の実装
def cross_validation_5_fold(tensors, labels, created_model_name="sample_5_fold"):
    
    LABELS = list(set(labels))
    
    k = 1
    margin1 = margin2 = 5.0
    beta = 1e-4
    dimension = tensors[0].size()[0]
    num_epoch = 300


    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results_error_rate = []
    fold_results_f1_score = []

    indexes = range(len(labels))
    
    for fold, (train_index, test_index) in enumerate(kf.split(indexes)):

        print(f"Fold {fold + 1}")
        train_tensors, test_tensors = [tensors[i] for i in train_index], [tensors[i] for i in test_index]
        train_labels, test_labels = [labels[i] for i in train_index], [labels[i] for i in test_index]

        ewt = 0
        ud = 0
        for label in test_labels:
            if label == "EWT":
                ewt += 1
            else :
                ud += 1
        print(ewt, ud)

        test_size = len(test_index)

        # 正例、負例ペアの生成
        positive, negative = create_pairs_lmnn(train_tensors, train_labels, weights=torch.ones(dimension), k=k)


        # クラスインスタンスの生成
        distance_function = WeightedPqgramDistance(dimension, positive, negative)
        distance_function = distance_function.to("cuda")

        criterion = MetricLearingLoss(margin1, margin2, beta).to("cuda")
    

            
        optimizer = optim.Adam(distance_function.parameters(), lr=0.001)

        train_loss_list = []

        model_path = "models/" + created_model_name + str(fold) + ".pth"
        
        # Training loop
        for epoch in tqdm(range(num_epoch), desc=f"[training]", leave=False):
            optimizer.zero_grad()
            loss = criterion(distance_function, distance_function.positive_pairs, distance_function.negative_pairs)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.detach().cpu().numpy())

            if epoch == 0:
                #print(f'\nEpoch: {epoch+1},\tLoss: {loss.item()}')
                best_epoch = epoch+1
                best_loss = loss.item()
                best_model = distance_function
                continue

            if loss.item() < best_loss:
                best_epoch = epoch+1
                best_loss = loss.item()
                best_model = distance_function

            if epoch%50 == 49:
                #print(f'\nEpoch: {epoch+1},\tLoss: {loss.item()}')
                distance_function.setPairs(train_tensors, train_labels, k)
        
        print(f"{best_epoch},{best_loss}\n")
        torch.save(best_model.state_dict(), model_path)
        
        # Evaluation

        weights = best_model.weights
        with torch.no_grad():

            M = [[0,0],[0,0]]

            train_tensors = torch.stack([t.to("cuda") for t in train_tensors])
            test_tensors = torch.stack([t.to("cuda") for t in test_tensors])

            for i in tqdm(range(test_size),desc="[test loop]"):

                test_tensor = test_tensors[i].unsqueeze(0)  # [1, dim]の形状に変換

                # 全ての訓練テンソルとの距離をバッチで計算
                distances = weighted_pqgram_distance_batch(weights, train_tensors, test_tensor.repeat(train_tensors.size(0), 1))
                print(len(distances))

                # 最小の距離を持つテンソルを見つける
                pred_id = torch.argmin(distances).item()
                pred_label = train_labels[pred_id]
                #print(pred_id, distances[pred_id], sorted(distances)[0])

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

            print(M[0][0]+M[0][1], M[1][0]+M[1][1])
            
            fold_results_error_rate.append((M[0][1]+M[1][0])/test_size)
            TP = M[0][0]/(M[0][0]+M[0][1])
            FP = M[0][0]/(M[0][0]+M[1][0])

            fold_results_f1_score.append(2*TP*FP/(TP+FP))

            print(M)
    
    # Cross-validation results
    #print(f"Average Accuracy: {np.mean(fold_results_error_rate) * 100:.2f}%")
    return fold_results_error_rate, fold_results_f1_score

# 実行例
# tensors, labels, weightsは適切なデータに置き換えて使用してください。
# modelも事前に定義されたニューラルネットワークモデルに置き換えてください。
# cross_validation_5_fold(tensors, labels, weights, model, epochs=10)
