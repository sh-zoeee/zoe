import numpy as np
from sklearn.model_selection import train_test_split
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

# weighted pq-gram distance の計算
def weighted_pqgram_distance(weights, tensor1: torch.Tensor, tensor2: torch.Tensor):
    device = tensor1.device
    min12 = torch.minimum(tensor1, tensor2).to(device)
    diff = tensor1+tensor2-2*min12
    aw = func.softplus(weights).to(device)
    return torch.dot(diff, aw)


# weighted pq-gram distance の距離行列の作成
def distance_matrix(tensors: list, weights):
    num_tensor = len(tensors)
    dist_mat = torch.zeros(num_tensor, num_tensor, dtype=torch.float32)
    for i in range(num_tensor):
        for j in range(i+1, num_tensor):
            dist = weighted_pqgram_distance(weights, tensors[i], tensors[j])
            dist_mat[i][j] = dist
            dist_mat[i][j] = dist
    return dist_mat


def create_pairs_lmnn(data, labels, weights, k):

    data_size_bound = 200

    if len(data)>data_size_bound:
        indexes = choices(range(len(data)), k=data_size_bound)

    data_picked = []
    labels_picked = []
    for id in indexes:
        data_picked.append(data[id])
        labels_picked.append(labels[id])

    distances = distance_matrix(data_picked, weights)

    positive_pairs = []
    negative_pairs = []

    for i in tqdm(range(data_size_bound), desc="[LMNN]"):

        distances_asc_arg = torch.argsort(distances[i])[1:]

        targets = []
        impostors = []
        label_i = labels_picked[i]
        j = 0
        while len(targets)<=k and j<199:
            idx = distances_asc_arg[j]
            if labels_picked[idx] == label_i:
                targets.append(idx)
            else:
                impostors.append(idx)
            j += 1
    
    positive_pairs.extend([(data_picked[i], data_picked[j]) for j in targets])
    negative_pairs.extend([(data_picked[i], data_picked[j]) for j in impostors])

    return positive_pairs, negative_pairs

# 
class WeightedPqgramDistance(nn.Module):
  
    def __init__(self, dimension):
        super(WeightedPqgramDistance, self).__init__()
        self.weights = nn.Parameter(torch.ones(dimension))
    
    def forward(self, tensor1, tensor2):
        dist = weighted_pqgram_distance(self.weights, tensor1, tensor2)
        return dist
  

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
        loss += self.beta * reg_term

        return loss


# k-NNによるラベル推測のための関数
"""
def predict_label_knn(model, test_tensor, train_tensors, train_labels, k):

    # 各訓練データとの距離を計算
    distances = []
    for tensor, label in zip(train_tensors, train_labels):
        distance = model(test_tensor, tensor).item()
        distances.append((distance, label))

    # 距離でソートし, k個の隣人を選択 
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]

    # 最も近いラベルを返す
    return Counter(k_nearest_labels).most_common(1)[0][0]
"""


# k-NNによるラベル推測のための関数
def predict_label_knn(weights, test_tensor, train_tensors, train_labels, k):

    # 各訓練データとの距離を計算
    distances = []
    for tensor, label in zip(train_tensors, train_labels):
        distance = weighted_pqgram_distance(weights, test_tensor, tensor)
        distances.append(distance)
    
    indexes = torch.topk(torch.tensor(distances), k, largest=False)[1]
    classes = [train_labels[i] for i in indexes]

    return Counter(classes).most_common(1)[0][0]


def predict_label_1nn(model, test_tensor: torch.Tensor, train_tensors, train_labels):

    # 各訓練データとの距離を計算
    min_dist = model(test_tensor, train_tensors[0].to("cuda:0")).item()
    min_idx = 0

    for i, tensor in enumerate(train_tensors[1:], start=1):
        distance = model(test_tensor, tensor.to("cuda:0")).item()
        if distance < min_dist:
            min_dist = distance
            min_idx = i

    return train_labels[min_idx]


"""
def predict_label_1nn(weights, test_tensor: torch.Tensor, train_tensors, train_labels):

    # 各訓練データとの距離を計算
    min_dist = weighted_pqgram_distance(weights.cuda(), test_tensor, train_tensors[0].cuda()).to(device="cpu")
    min_idx = 0

    for i, tensor in enumerate(train_tensors[1:], start=1):
        distance = weighted_pqgram_distance(weights.cuda(), test_tensor, tensor.cuda()).to(device="cpu")
        if distance < min_dist:
            min_dist = distance
            min_idx = i

    return train_labels[min_idx]
"""



# 外部からは以下を呼び出す 
# データの前処理 
def preprocessing(CORPUS_FILE: list, type_of_labels: list, save_file_list: list):
    CORPUS_LIST = []
    for corpus in CORPUS_FILE:
        CORPUS_LIST.append(corpus.split(".")[0].split("/")[-1])

    CORPUS_i_LENGTH = []

    CoNLL = []
    labels = []
    for i in range(len(CORPUS_FILE)):
        tmp_conll = pyconll.load_from_file(CORPUS_FILE[i])
        CoNLL += tmp_conll
        for _ in range(len(tmp_conll)):
            labels.append(type_of_labels[i])
        if i != 0:
            CORPUS_i_LENGTH.append(len(tmp_conll)+CORPUS_i_LENGTH[i-1])
        else:
            CORPUS_i_LENGTH.append(len(tmp_conll))
    
    print(CORPUS_i_LENGTH)
    
    num_trees = CORPUS_i_LENGTH[-1]
    #pqtrees = [trees.conllTree_to_pqTree_upos(conll.to_tree()) for conll in CoNLL]
    pqtrees = [trees.conllTree_to_pqTree_unlabeled(conll.to_tree()) for conll in CoNLL]
    
    pqIndex = [Profile(tree, p=2, q=2) for tree in pqtrees]

    J = set(pqIndex[0])
    for pq_set in pqIndex[1:]:
        J = J.union(pq_set)
    J = list(J)

    print(f'vector dimension: {len(J)}')

    tensors = [pq_gram.pqgram_to_tensor(pqgram, J) for pqgram in tqdm(pqIndex, desc="[convert tensor]")]
    
    indexes = torch.Tensor(range(num_trees))

    # 訓練データとテストデータに分割
    train_tensors, test_tensors, train_labels, test_labels, train_indexes, test_indexes = train_test_split(tensors, labels, indexes, test_size=0.4, random_state=10) # 無印はrandom state = 42
    valid_tensors, test_tensors, valid_labels, test_labels, valid_indexes, test_indexes = train_test_split(test_tensors, test_labels, test_indexes, test_size=0.5, random_state=10)


    # データの保存
    torch.save(train_tensors, save_file_list[0])
    torch.save(train_labels, save_file_list[1])
    torch.save(train_indexes, save_file_list[2])

    torch.save(valid_tensors, save_file_list[3])
    torch.save(valid_labels, save_file_list[4])
    torch.save(valid_indexes, save_file_list[5])

    torch.save(test_tensors, save_file_list[6])    
    torch.save(test_labels, save_file_list[7])
    torch.save(test_indexes, save_file_list[8])


def train(
        train_tensors_file: str, train_labels_file: str,
        valid_tensors_file: str, valid_labels_file: str,
        created_model_path: str, loss_figure_path='figures/loss.png'
        ):

    # 訓練データのロード
    train_tensors = torch.load(train_tensors_file)
    train_labels = torch.load(train_labels_file)

    # 検証データのロード
    valid_tensors = torch.load(valid_tensors_file)
    valid_labels = torch.load(valid_labels_file)

    # ハイパーパラメータなど
    k = 1
    margin1 = margin2 = 5.0
    beta = 1e-4
    dimension = train_tensors[0].size()[0]
    num_epoch = 2000

    # クラスインスタンスの生成
    distance_function = WeightedPqgramDistance(dimension)
    criterion = MetricLearingLoss(margin1, margin2, beta)

    # オプティマイザ
    optimizer = optim.Adam(distance_function.parameters(), lr=0.01)
    
    # 正例、負例ペアの生成
    positive, negative = create_pairs_lmnn(train_tensors, train_labels, weights=torch.ones(dimension), k=k)
    positive_val, negative_val = create_pairs_lmnn(valid_tensors, valid_labels, weights=torch.ones(dimension), k=k)

    # 各エポックの損失の合計値
    train_loss_list = []
    valid_loss_list = []

    # トレーニングループ 
    for epoch in tqdm(range(num_epoch), desc="[train loop]"):
        
        optimizer.zero_grad()
        loss = criterion(distance_function, positive, negative)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.detach().numpy())

        with torch.no_grad():
            val_loss = criterion(distance_function, positive_val, negative_val)
            valid_loss_list.append(val_loss.detach().numpy())

        if epoch == 0:
            print(f'\nEpoch: {epoch+1},\tLoss: {val_loss.item()}')
            best_epoch = epoch+1
            best_loss = val_loss.item()
            torch.save(distance_function.state_dict(), created_model_path)
            continue

        if val_loss.item() < best_loss:
            best_epoch = epoch+1
            best_loss = val_loss.item()
            torch.save(distance_function.state_dict(), created_model_path)

        if epoch%50 == 49:
            print(f'\nEpoch: {epoch+1},\tLoss: {val_loss.item()}')
            positive, negative = create_pairs_lmnn(train_tensors, train_labels, distance_function.weights, k)
        
    print(f'\nBest Epoch: {best_epoch},\tLoss: {best_loss}')

    plt.clf()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(num_epoch), train_loss_list, label='train')
    plt.plot(range(num_epoch), valid_loss_list, label='valid')
    plt.legend()
    plt.grid()
    plt.savefig(loss_figure_path)


def test(
        train_tensors_file: str, train_labels_file: str,
        test_tensors_file: str, test_labels_file: str,
        model_path: str
        ):
    
    train_tensors = torch.load(train_tensors_file)
    train_labels = torch.load(train_labels_file)

    test_tensors = torch.load(test_tensors_file)
    test_labels = torch.load(test_labels_file)

    dimension_tensor = test_tensors[0].size()[0]

    # モデルと損失関数 
    distance_function = WeightedPqgramDistance(dimension_tensor)
    distance_function.load_state_dict(torch.load(model_path))
    distance_function.eval().to("cuda:0")
    

    train_tensors, _, train_labels, _= train_test_split(train_tensors, train_labels, random_state=42, shuffle=True, train_size=2000)

    num_test = len(test_labels)
    error = 0

    k = 1
    
    for i in tqdm(range(num_test), desc="[test  loop]"):
        
        test_tensor = test_tensors[i].to("cuda:0")
        
        predicted_label = predict_label_1nn(
            distance_function, test_tensor, train_tensors, train_labels
        )
        
        if predicted_label != test_labels[i]:
            error += 1
    
    print(f'\terror rate: {error/num_test:.2f}')

