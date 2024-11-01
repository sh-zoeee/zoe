import numpy as np
from pqgrams.PQGram import Profile
from tqdm import tqdm
from zss import simple_distance
import torch 


def count_nodes(tree) -> int:
    count = 0
    for _ in tree.iter():
        count += 1
    return count


def N_TED(tree1, tree2, alpha=1):
    ted = simple_distance(tree1, tree2)
    size1 = count_nodes(tree1)
    size2 = count_nodes(tree2)
    return (2*ted)/(alpha*(size1+size2)+ted)


def pqgram_distance(pqgram1: Profile, pqgram2: Profile, normalize=False) -> int:
    """
    pqgram1: pqgram profile
    pqgram2: pqgram profile
    """
    union = len(pqgram1) + len(pqgram2)
    intersection = pqgram1.intersection(pqgram2)
    dist = union - 2*intersection
    if not normalize:
        return dist
    else:
        return dist/(union-intersection)


def pqgram_distance_tensor(tensor1: torch.Tensor, tensor2: torch.Tensor):

    device = tensor1.device
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(device=device)
    
    dim = tensor1.size()[0]

    tensor_min = torch.minimum(tensor1, tensor2).to(device)
    tensor_diff = tensor1 + tensor2 - 2*tensor_min

    d = torch.dot(torch.ones(dim, dtype=torch.float32, device=device),tensor_diff)
    return float(d.to("cpu").detach().numpy())



def distance_matrix_pq(pqgram_list: list, nums_text) -> np.ndarray:
    """
    pqgram_list: list of pqgram profiles
    """
    distance_matrix = np.zeros((nums_text, nums_text), dtype=int)
    for i in tqdm(range(nums_text)):
        for j in range(i+1, nums_text):
            dist = pqgram_distance(pqgram_list[i], pqgram_list[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix


def distance_matrix_ted(nodes: list, nums_text) -> np.ndarray:
    """
    nodes: list of zss.Node
    """
    distance_matrix = np.zeros((nums_text, nums_text), dtype=int)
    for i in tqdm(range(nums_text)):
        for j in range(i+1, nums_text):
            dist = simple_distance(nodes[i], nodes[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    return distance_matrix



