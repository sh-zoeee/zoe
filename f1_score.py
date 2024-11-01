from scripts import w_pq_batch as w_pq
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time, torch
from tqdm import tqdm

def main():

    train_tensors_path = "data/train_tensors_en_corpora_EWT_udpipe_unlabel_10.pt"
    train_labels_path = "data/train_labels_en_corpora_EWT_udpipe_unlabel_10.pt"

    test_tensors_path = "data/test_tensors_en_corpora_EWT_udpipe_unlabel_10.pt"
    test_labels_path = "data/test_labels_en_corpora_EWT_udpipe_unlabel_10.pt"

    model_path = "models/model_en_corpora_EWT_udpipe_unlabel_10.pth"

    

    train_tensors = torch.load(train_tensors_path)
    train_tensors = torch.stack([t.to("cuda") for t in train_tensors])

    test_tensors = torch.load(test_tensors_path)
    test_tensors = torch.stack([t.to("cuda") for t in test_tensors])

    train_labels = torch.load(train_labels_path)

    test_labels = torch.load(test_labels_path)

    LABELS = list(set(train_labels))

    dimension_tensor = test_tensors[0].size()[0]

    # モデルと損失関数 
    distance_function = w_pq.WeightedPqgramDistance(dimension_tensor, [], [])
    distance_function.load_state_dict(torch.load(model_path))
    distance_function.eval().to("cuda:0")
    

    test_size = len(test_labels)  # テストデータのサイズ
    error = 0

    M = [[0,0],[0,0]]

    k = 1  # k-NNのk値

    
    for i in tqdm(range(test_size),desc="[test loop]"):

        test_tensor = test_tensors[i].unsqueeze(0)  # [1, dim]の形状に変換

        # 全ての訓練テンソルとの距離をバッチで計算
        distances = w_pq.weighted_pqgram_distance_batch(distance_function.weights, train_tensors, test_tensor.repeat(train_tensors.size(0), 1))

        # 最小の距離を持つテンソルを見つける
        pred_id = torch.argmin(distances).item()
        pred_label = train_labels[pred_id]

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

    print(M[0][1]+M[1][0], error)

    print(f'\terror rate: {error / test_size:.2f}\n')

    TP = M[0][0]/(M[0][0]+M[0][1])
    FP = M[0][0]/(M[0][0]+M[1][0])

    print(f'\tf1 score: {2*TP*FP/(TP+FP):.3f}\n')
    
    return 


if __name__ == "__main__":
    main()
