from scripts import w_pq_batch as w_pq


import numpy as np
from sklearn.model_selection import train_test_split
import pyconll
from pqgrams.PQGram import Profile
from scripts import pq_gram, trees, func
import torch
from torch import nn, optim
from torch.nn import DataParallel
from random import choices
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from time import time
from os import mkdir, makedirs, path


from statistics import mode


def main():
    
    random_state = 10
    label_type = "unlabel"
    p = 2
    q = 2
    print(label_type, end="\t", flush=True)

    torch.cuda.empty_cache()

    corpora = list()
    corpora.append(f"corpora/English/English-ESL.conllu")
    corpora.append(f"corpora/English/English-Atis.conllu")
    #corpora.append(f"parsed/En-EWT-spacy.conllu")
    corpora.sort()
    CORPORA = corpora
    
    
    text = CORPORA[0].split("/")[-1].split(".")[0] + "_"
    print(text[:-1], end=" vs ", flush=True)
    dir_name = text
    text = CORPORA[1].split("/")[-1].split(".")[0]
    print(text, end="\n", flush=True)
    dir_name += text


    DIR_NAME = f"data_cross/{label_type}/" + dir_name + "/"
    MODEL_DIR = DIR_NAME+"models/"
    if not path.exists(MODEL_DIR):
        mkdir(MODEL_DIR)

    tensors_list = list()
    labels_list = list()
    indexes_list = list()

    for i in range(1,6):
        tensors_list.append(torch.load(DIR_NAME+f"tensor_{i}.pt"))
        labels_list.append(torch.load(DIR_NAME+f"labels_{i}.pt"))
        indexes_list.append(torch.load(DIR_NAME+f"indexes_{i}.pt"))
    
    num_fold = len(labels_list)

    # ハイパーパラメータなど
    k = 1
    margin1 = margin2 = 5.0
    beta = 1e-4
    dimension = tensors_list[0][0].size()[0]
    num_epoch = 1000

    error_rate_list = []
    f1_score_list = []
    time_list = []
    
    for fold in range(num_fold):
        print(f"\nfold {1+fold}/{num_fold}: ")

        train_tensors = tensors_list.copy()
        valid_tensors = train_tensors.pop(fold)
        test_tensors = train_tensors.pop(fold%len(train_tensors))

        train_tensors = sum(train_tensors, [])

        train_tensors = torch.stack([t.to("cuda:1") for t in train_tensors])
        valid_tensors = torch.stack([t.to("cuda:1") for t in valid_tensors])

        train_labels = labels_list.copy()
        valid_labels = train_labels.pop(fold)
        test_labels = train_labels.pop(fold%len(train_labels))

        train_labels = sum(train_labels, [])
        
        start = time()



        # 正例、負例ペアの生成
        positive, negative = w_pq.create_pairs_lmnn(train_tensors, train_labels, weights=torch.ones(dimension), k=k)
        positive_val, negative_val = w_pq.create_pairs_lmnn(valid_tensors, valid_labels, weights=torch.ones(dimension), k=k)

        # クラスインスタンスの生成
        distance_function = w_pq.WeightedPqgramDistance(dimension, positive, negative)
        distance_function = distance_function.to("cuda:1")

        criterion = w_pq.MetricLearingLoss(margin1, margin2, beta).to("cuda:1")

        # オプティマイザ
        optimizer = optim.Adam(distance_function.parameters(), lr=0.01)
    
        # 各エポックの損失の合計値
        train_loss_list = []
        valid_loss_list = []

        model_path = MODEL_DIR+f"model_{fold}-{num_fold}.pth"

        for epoch in tqdm(range(num_epoch), desc=f"[train loop {1+fold}]", leave=False):
            
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
                torch.save(distance_function.state_dict(), model_path)
                continue

            if val_loss.item() < best_loss:
                best_epoch = epoch+1
                best_loss = val_loss.item()
                torch.save(distance_function.state_dict(), model_path)

            if epoch%50 == 49:
                print(f'\nEpoch: {epoch+1},\tLoss: {val_loss.item()}')
                distance_function.setPairs(train_tensors, train_labels, k)

        # test()
        train_tensors = torch.stack([t.to("cuda:0") for t in train_tensors])
        test_tensors = torch.stack([t.to("cuda:0") for t in test_tensors])


        LABELS = list(set(test_labels))

        dimension_tensor = test_tensors[0].size()[0]

        # モデルと損失関数 
        distance_function = w_pq.WeightedPqgramDistance(dimension_tensor, [], [])
        distance_function.load_state_dict(torch.load(model_path))
        distance_function.eval().to("cuda:0")
        

        test_size = len(test_labels)  # テストデータのサイズ

        M = [[0,0],[0,0]] # 混同行列

        k = 1  # k-NNのk値

        
        for i in tqdm(range(test_size),desc=f"[ test loop {1+fold}]", leave=False):

            test_tensor = test_tensors[i].unsqueeze(0)  # [1, dim]の形状に変換

            # 全ての訓練テンソルとの距離をバッチで計算
            distances = w_pq.weighted_pqgram_distance_batch(distance_function.weights, train_tensors, test_tensor.repeat(train_tensors.size(0), 1))

            # 最小の距離を持つテンソルを見つける

            pred_id = torch.argmin(distances).item()

            if distances[pred_id]==0:
                zero_dists = torch.nonzero(distances == 0, as_tuple=False).to("cpu").detach().numpy()
                zero_id = np.ndarray(len(zero_dists), dtype=int)
                for i, id in enumerate(zero_dists):
                    zero_id[i] = int(id.item())
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

        end = time()

        error_rate = (M[0][1]+M[1][0])/test_size

        TP = M[0][0]/(M[0][0]+M[0][1])
        FP = M[0][0]/(M[0][0]+M[1][0])

        f1_score = 2*TP*FP/(TP+FP)

        error_rate_list.append(error_rate)
        f1_score_list.append(f1_score)
        time_list.append(end-start)

        print(f"fold {fold}: {end-start}sec")

    print("ERROR RATE: ",error_rate_list)
    print("\t", "average: ", np.mean(error_rate_list), "standard diviation: ", np.std(error_rate_list))
    print("F1 SCORE: ", f1_score_list)
    print("\t", "average: ", np.mean(f1_score_list), "standard diviation: ", np.std(f1_score_list))
    print("time: ", time_list)
    print("\t", "average: ", np.mean(time_list), "standard diviation: ", np.std(time_list))

    

if __name__=="__main__":
    main()

