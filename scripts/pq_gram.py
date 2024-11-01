from pqgrams.PQGram import Profile
import torch
from collections import Counter

def pqgram_to_tensor(pqgrams: Profile, J: list) -> torch.Tensor:
    dim = len(J)
    vec = torch.zeros(dim, dtype=torch.float32)
    pqgrams_set = list(set(pqgrams))
    counts = Counter(pqgrams)
    for pqgram in pqgrams_set:
      count = counts[pqgram]
      idx = J.index(pqgram)
      vec[idx] = count
    return vec