import pyconll
from scripts import trees, pq_gram
from pqgrams.PQGram import Profile 

from tqdm import tqdm

import torch


def main():

    upos_or_unlabel = True # Trueならupos、Falseならunlabel
    node_label = "unlabel"

    if upos_or_unlabel:
        node_label = "upos"

    p = q = 2
    CONLLUs = ["EWT", "EWT"]

    CONLL_FILES = [f"corpora/English/English-{CONLLUs[0]}.conllu", f"corpora/English/English-{CONLLUs[1]}.conllu"]
    TENSOR_FILE = f"data/all_tensors/{CONLLUs[0]}-{CONLLUs[1]}_{node_label}.pt"
    
    CoNLL = pyconll.load_from_file(CONLL_FILES[0])
    CoNLL += pyconll.load_from_file(CONLL_FILES[1])

    pqtrees = [trees.conllTree_to_pqTree_upos(conll.to_tree()) for conll in CoNLL]

    pqIndex = [Profile(tree, p=2, q=2) for tree in pqtrees]

    J = set(pqIndex[0])
    for pq_set in pqIndex[1:]:
        J = J.union(pq_set)
    J = list(J)

    tensors = [pq_gram.pqgram_to_tensor(pqgram, J) for pqgram in tqdm(pqIndex, desc="[convert tensor]")]
    
    torch.save(tensors, )


    return 


if __name__ == "__main__":
    main()
