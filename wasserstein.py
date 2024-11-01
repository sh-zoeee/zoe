

from scripts import w_pq_batch as w_pq
from scripts import pq_gram, trees, func

import time, torch, random, pyconll, ot
import numpy as np

from pqgrams.PQGram import Profile

from tqdm import tqdm

import matplotlib.pyplot as plt

def cost_matrix(tensors, weights, chunk_size):
    device = "cuda:3"

    num_samples = tensors.shape[0]
    dist_mat = torch.zeros((num_samples, num_samples), device=device)  # 距離行列の初期化
    
    # チャンクごとに計算
    for i in range(0, num_samples, chunk_size):
        end_i = min(i + chunk_size, num_samples)
        tensor_chunk_i = tensors[i:end_i].unsqueeze(1)  # [chunk_size, 1, dim]

        for j in range(0, num_samples, chunk_size):
            end_j = min(j + chunk_size, num_samples)
            tensor_chunk_j = tensors[j:end_j].unsqueeze(0)  # [1, chunk_size, dim]
            
            # 差の計算
            diff = torch.abs(tensor_chunk_i - tensor_chunk_j).to(device)  # [chunk_size, chunk_size, dim]
            aw = func.softplus(weights).to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
            weighted_diff = diff * aw  # アダマール積
            dist_chunk = weighted_diff.sum(dim=2)  # 距離の計算
            
            # 距離行列に結果を格納
            dist_mat[i:end_i, j:end_j] = dist_chunk
        del dist_chunk, end_j, aw, weighted_diff
        torch.cuda.empty_cache()
    
    del tensors, end_i
    torch.cuda.empty_cache()

    return dist_mat




def main():

    torch.cuda.empty_cache()

    train_tensors_path = "data/train_tensors_en_corpora_En_EWT_chatGPT_unlabel_50.pt"
    train_labels_path = "data/train_labels_en_corpora_En_EWT_chatGPT_unlabel_50.pt"
    train_indexes_path = "data/train_indexes_en_corpora_En_EWT_chatGPT_unlabel_50.pt"

    valid_tensors_path = "data/valid_tensors_en_corpora_En_EWT_chatGPT_unlabel_50.pt"
    valid_labels_path = "data/valid_labels_en_corpora_En_EWT_chatGPT_unlabel_50.pt"
    valid_indexes_path = "data/valid_indexes_en_corpora_En_EWT_chatGPT_unlabel_50.pt"

    test_tensors_path = "data/test_tensors_en_corpora_En_EWT_chatGPT_unlabel_50.pt"
    test_labels_path = "data/test_labels_en_corpora_En_EWT_chatGPT_unlabel_50.pt"
    test_indexes_path = "data/test_indexes_en_corpora_En_EWT_chatGPT_unlabel_50.pt"

    model_path = "models/model_en_corpora_EWT_Atis_unlabel_50.pth"


    CoNLLU_EWT_PATH = "corpora/English-EWT.conllu"
    CoNLLU_GPT_PATH = "corpora/English-Atis.conllu"
    CoNLLU = pyconll.load_from_file(CoNLLU_EWT_PATH)
    EWT_tree_count = len(CoNLLU)
    CoNLLU += pyconll.load_from_file(CoNLLU_GPT_PATH)
    GPT_tree_count = len(CoNLLU)-EWT_tree_count
    

    PQ_Trees = [trees.conllTree_to_pqTree_unlabeled(conll.to_tree()) for conll in CoNLLU]
    PQ_Index = [Profile(tree, p=2, q=2) for tree in PQ_Trees]

    J = set(PQ_Index[0])
    for pq_set  in PQ_Index[1:]:
        J = J.union(pq_set)
    J = list(J)

    tensors = [pq_gram.pqgram_to_tensor(pqgram, J) for pqgram in tqdm(PQ_Index, desc="[convert tensor]")]

    distance_function = w_pq.WeightedPqgramDistance(tensors[0].size(), [], [])
    distance_function.load_state_dict(torch.load(model_path))
    distance_function.eval()

    weights = distance_function.weights

    tensors_EWT = tensors[:EWT_tree_count]
    tensors_GPT = tensors[EWT_tree_count:]
    sample_size = 100#min(EWT_tree_count, len(tensors)-EWT_tree_count)
    if EWT_tree_count > sample_size:
        tensors_GPT = random.sample(tensors_EWT, k=sample_size)
    if GPT_tree_count > sample_size:
        tensors_EWT = random.sample(tensors_GPT, k=sample_size)
    
    Tensors_EWT = torch.zeros((EWT_tree_count,8))
    for i, tensor in enumerate(tensors_EWT[:100]):
        Tensors_EWT[i] = tensor

    Tensors_GPT = torch.zeros((len(tensors_GPT),8))
    for i, tensor in enumerate(tensors_GPT[:100]):
        Tensors_GPT[i] = tensor

    array_EWT = Tensors_EWT.numpy()
    array_GPT = Tensors_GPT.numpy()

    weights_np = weights.detach().numpy()

    array_size = len(array_EWT) + len(array_GPT)

    array_all = np.zeros((200, 8))
    
    for i in range(100):
        array_all[i] = array_EWT[i]

    for i in range(100):
        array_all[i+100] = array_GPT[i]

    print(len(array_all))

    distance_matrix = torch.zeros((200,200))

    tensors = torch.from_numpy(array_all)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    chunk_size = 4
    for i in range(0, 200, chunk_size):
        end_i = min(i + chunk_size, 200)
        tensor_chunk_i = tensors[i:end_i].unsqueeze(1)  # [chunk_size, 1, dim]

        for j in range(0, 200, chunk_size):
            end_j = min(j + chunk_size, 200)
            tensor_chunk_j = tensors[j:end_j].unsqueeze(0)  # [1, chunk_size, dim]
            
            # 差の計算
            diff = torch.abs(tensor_chunk_i - tensor_chunk_j).to(device)  # [chunk_size, chunk_size, dim]
            aw = func.softplus(weights).to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
            weighted_diff = diff * aw  # アダマール積
            dist_chunk = weighted_diff.sum(dim=2)  # 距離の計算
            
            # 距離行列に結果を格納
            distance_matrix[i:end_i, j:end_j] = dist_chunk
        del dist_chunk, end_j, aw, weighted_diff
        torch.cuda.empty_cache()
    
    del tensors, end_i
    torch.cuda.empty_cache()

    distance_matrix /= distance_matrix.max()

    w_norm = np.ones((200,))/200


    print(distance_matrix.detach().numpy())

    T = ot.emd2(w_norm, w_norm, distance_matrix.to("cpu").detach().numpy())
    print(T)


if __name__=="__main__":
    main()

