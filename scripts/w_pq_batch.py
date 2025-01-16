import numpy as np
from sklearn.model_selection import train_test_split
import pyconll
from pqgrams.PQGram import Profile
from . import pq_gram, trees, func
import torch
from torch import nn, optim
from torch.nn import DataParallel
from random import choices
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from time import time

from statistics import mode




# weighted pq-gram distance の計算
def weighted_pqgram_distance(weights, tensor1: torch.Tensor, tensor2: torch.Tensor):
    device = tensor1.device
    min12 = torch.minimum(tensor1, tensor2).to(device)
    diff = tensor1+tensor2-2*min12
    aw = func.softplus(weights.to(device))
    return torch.dot(diff,aw)


def weighted_pqgram_distance_batch(weights, batch1: torch.Tensor, batch2: torch.Tensor):
    assert batch1.device == batch2.device
    assert batch1.shape == batch2.shape

    device = batch1.device
    min12 = torch.minimum(batch1, batch2)
    diff = batch1+batch2-2*min12
    aw = func.softplus(weights.to(device))
    return (diff*aw).sum(dim=1)


def normalize_tensor(t: torch.Tensor) -> torch.Tensor:

    min_val = t.min(dim=0).values
    max_val = t.max(dim=0).values

    return (t-max_val)/(max_val-min_val)


def standardize_tensor(t: torch.Tensor) -> torch.Tensor:

    mean = t.mean(dim=0)
    std = t.std(dim=0)
    
    return (t-mean)/std



# weighted pq-gram distance の距離行列の作成
def distance_matrix(tensors: torch.Tensor, weights: torch.Tensor):

    device = weights.device

    tensor_stack = torch.stack(tensors).to(device) # テンソルをスタックして [N, dim] の形状に
    diff = tensor_stack.unsqueeze(1) - tensor_stack.unsqueeze(0)  # [N, N, dim] となるように展開
    diff = torch.abs(diff)  # 差の絶対値をとる
    aw = func.softplus(weights).to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, dim] としてアダマール積を取る
    weighted_diff = diff * aw  # アダマール積!
    dist_mat = weighted_diff.sum(dim=2)  # 次元ごとの合計を取ることで距離行列が得られる
    return dist_mat


def distance_matrix_chunked(tensors: torch.Tensor, weights: torch.Tensor, chunk_size: int):
    """
    データをチャンクに分割して距離行列を計算する関数。
    tensors: 入力データ [N, dim] のテンソル
    weights: 重み [dim] のテンソル
    chunk_size: 一度に処理するデータのチャンクサイズ
    """
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    num_samples = tensors.shape[0]
    dist_mat = torch.zeros((num_samples, num_samples), device=device)  # 距離行列の初期化
    aw = func.softplus(weights).unsqueeze(0).unsqueeze(0)  # [1, 1, dim]

    # チャンクごとに計算
    for i in tqdm(range(0, num_samples, chunk_size), desc="distance matrix", leave=False):
        end_i = min(i + chunk_size, num_samples)
        tensor_chunk_i = tensors[i:end_i].unsqueeze(1)  # [chunk_size, 1, dim]

        for j in range(0, num_samples, chunk_size):
            end_j = min(j + chunk_size, num_samples)
            tensor_chunk_j = tensors[j:end_j].unsqueeze(0)  # [1, chunk_size, dim]
            
            # 差の計算
            diff = torch.abs(tensor_chunk_i - tensor_chunk_j).to(device)  # [chunk_size, chunk_size, dim]
            
            weighted_diff = diff * aw.to(device)  # アダマール積
            dist_chunk = weighted_diff.sum(dim=2)  # 距離の計算
            
            # 距離行列に結果を格納
            dist_mat[i:end_i, j:end_j] = dist_chunk
        del dist_chunk, end_j, weighted_diff
        torch.cuda.empty_cache()
    
    del tensors, end_i, aw
    torch.cuda.empty_cache()

    return dist_mat



def create_pairs_lmnn(data, labels, weights, k):

    #distances = distance_matrix(data, weights)
    upper_bound = min(len(data),1000)
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
def predict_label_knn(weights, test_tensors, train_tensors, train_labels, k):
    # 各テストデータの距離を保存するリスト
    predicted_labels = []

    # バッチごとに処理
    for test_tensor in test_tensors:
        # 各訓練データとの距離を計算
        distances = []
        for tensor, label in zip(train_tensors, train_labels):
            distance = weighted_pqgram_distance(weights, test_tensor, tensor)
            distances.append(distance)

        # k近傍のインデックスを取得
        indexes = torch.topk(torch.tensor(distances), k, largest=False)[1]
        classes = [train_labels[i] for i in indexes]

        # 最も多いラベルを予測
        predicted_labels.append(Counter(classes).most_common(1)[0][0])

    return predicted_labels


def predict_label_1nn(model, test_tensor: torch.Tensor, train_tensors, train_labels):

    # 各訓練データとの距離を計算
    min_dist = model(test_tensor, train_tensors[0].to("cuda:2")).item()
    min_idx = 0

    for i, tensor in enumerate(train_tensors[1:], start=1):
        distance = model(test_tensor, tensor.to("cuda:2")).item()
        if distance < min_dist:
            min_dist = distance
            min_idx = i

    return train_labels[min_idx]



# 外部からは以下を呼び出す 
# データの前処理 
def preprocessing(
        CORPUS_FILE: list, 
        type_of_labels: list, 
        save_file_list: list,
        random_state : int = 89,
        p : int = 2,
        q : int = 2,
        label_type : str="unlabel",
        normalize : bool = False, 
        standardize : bool = False,
    ):
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
    if label_type == "upos":
        pqtrees = [trees.conllTree_to_pqTree_upos(conll.to_tree()) for conll in CoNLL]
    else:
        pqtrees = [trees.conllTree_to_pqTree_unlabeled(conll.to_tree()) for conll in CoNLL]
    
    pqIndex = [Profile(tree, p=p, q=q) for tree in pqtrees]

    J = set(pqIndex[0])
    for pq_set in pqIndex[1:]:
        J = J.union(pq_set)
    J = list(J)

    print(f'vector dimension: {len(J)}')

    tensors = [pq_gram.pqgram_to_tensor(pqgram, J) for pqgram in tqdm(pqIndex, desc="[convert tensor]")]

    if normalize:
        tensors = [normalize_tensor(tensor) for tensor in tensors]
    elif standardize:
        tensors = [standardize_tensor(tensor) for tensor in tensors]
    
    indexes = torch.Tensor(range(num_trees))

    # 訓練データとテストデータに分割
    train_tensors, test_tensors, train_labels, test_labels, train_indexes, test_indexes = train_test_split(tensors, labels, indexes, test_size=0.4, random_state=random_state) # 無印はrandom state = 42
    valid_tensors, test_tensors, valid_labels, test_labels, valid_indexes, test_indexes = train_test_split(test_tensors, test_labels, test_indexes, test_size=0.5, random_state=random_state)


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
    train_tensors = torch.stack([t.to("cuda:1") for t in train_tensors])
    train_labels = torch.load(train_labels_file)

    # 検証データのロード
    valid_tensors = torch.load(valid_tensors_file)
    valid_tensors = torch.stack([t.to("cuda:1") for t in valid_tensors])
    valid_labels = torch.load(valid_labels_file)

    # ハイパーパラメータなど
    k = 1
    margin1 = margin2 = 5.0
    beta = 1e-4
    dimension = train_tensors[0].size()[0]
    num_epoch = 1000

    # 正例、負例ペアの生成
    positive, negative = create_pairs_lmnn(train_tensors, train_labels, weights=torch.ones(dimension), k=k)
    positive_val, negative_val = create_pairs_lmnn(valid_tensors, valid_labels, weights=torch.ones(dimension), k=k)

    # クラスインスタンスの生成
    distance_function = WeightedPqgramDistance(dimension, positive, negative)
    distance_function = distance_function.to("cuda:1")

    criterion = MetricLearingLoss(margin1, margin2, beta).to("cuda:1")

    # オプティマイザ
    optimizer = optim.Adam(distance_function.parameters(), lr=0.01)
    


    # 各エポックの損失の合計値
    train_loss_list = []
    valid_loss_list = []

    # トレーニングループ 
    for epoch in tqdm(range(num_epoch), desc="[train loop]"):
        
        optimizer.zero_grad()
        loss = criterion(distance_function, distance_function.positive_pairs, distance_function.negative_pairs)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.detach().cpu().numpy())

        with torch.no_grad():
            val_loss = criterion(distance_function, positive_val, negative_val)
            valid_loss_list.append(val_loss.detach().cpu().numpy())

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
            distance_function.setPairs(train_tensors, train_labels, k)
        
    print(f'\nweights: {distance_function.weights}\n')
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
        model_path: str, batch_size: int = 64
        ):
    
    train_tensors = torch.load(train_tensors_file)
    train_tensors = torch.stack([t.to("cuda:0") for t in train_tensors])

    test_tensors = torch.load(test_tensors_file)
    test_tensors = torch.stack([t.to("cuda:0") for t in test_tensors])

    train_labels = torch.load(train_labels_file)

    test_labels = torch.load(test_labels_file)

    LABELS = list(set(test_labels))

    dimension_tensor = test_tensors[0].size()[0]

    # モデルと損失関数 
    distance_function = WeightedPqgramDistance(dimension_tensor, [], [])
    distance_function.load_state_dict(torch.load(model_path))
    distance_function.eval().to("cuda:0")
    

    test_size = len(test_labels)  # テストデータのサイズ

    M = [[0,0],[0,0]] # 混同行列

    k = 1  # k-NNのk値

    
    for i in tqdm(range(test_size),desc="[test loop]"):

        test_tensor = test_tensors[i].unsqueeze(0)  # [1, dim]の形状に変換

        # 全ての訓練テンソルとの距離をバッチで計算
        distances = weighted_pqgram_distance_batch(distance_function.weights, train_tensors, test_tensor.repeat(train_tensors.size(0), 1))

        # 最小の距離を持つテンソルを見つける
        """
        id_list = torch.argsort(distances, descending=False)[:k]
        label_list = [train_labels[id] for id in id_list]
        pred_label = mode(label_list)
        """
        
        

        pred_id = torch.argmin(distances).item()

        if distances[pred_id]==0:
            zero_dists = torch.nonzero(distances == 0, as_tuple=False).to("cpu").detach().numpy()
            zero_id = np.ndarray(len(zero_dists), dtype=int)
            for i, id in enumerate(zero_dists):
                zero_id[i] = int(id)
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

    print(f"{M}")

    print(f"混同行列 M: {M}")

    print(f'\terror rate: {(M[0][1]+M[1][0]) / test_size:.2f}')
    
    TP = M[0][0]/(M[0][0]+M[0][1])
    FP = M[0][0]/(M[0][0]+M[1][0])

    print(f'\tf1 score: {2*TP*FP/(TP+FP):.3f}\n')

    return
    

