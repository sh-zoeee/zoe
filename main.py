from scripts import w_pq_kfold as w_pq
from scripts import trees, pq_gram

import time, torch
from tqdm import tqdm

import pyconll

from pqgrams.PQGram import Profile



def main():

    random_state = 20

    torch.cuda.empty_cache()
    start = time.time()

    CORPORA =   ["corpora/English-EWT.conllu", "parsed/En-EWT-udpipe.conllu"]
    LABELS = ["EWT", "udpipe"]


    CORPUS_i_LENGTH = []

    CoNLL = []
    labels = []
    for i in range(len(CORPORA)):
        tmp_conll = pyconll.load_from_file(CORPORA[i])
        CoNLL += tmp_conll
        for _ in range(len(tmp_conll)):
            labels.append(LABELS[i])
        if i != 0:
            CORPUS_i_LENGTH.append(len(tmp_conll)+CORPUS_i_LENGTH[i-1])
        else:
            CORPUS_i_LENGTH.append(len(tmp_conll))
    
    print(CORPUS_i_LENGTH)
    
    num_trees = CORPUS_i_LENGTH[-1]
    pqtrees = [trees.conllTree_to_pqTree_unlabeled(conll.to_tree()) for conll in CoNLL]
    
    pqIndex = [Profile(tree, p=2, q=2) for tree in pqtrees]

    J = set(pqIndex[0])
    for pq_set in pqIndex[1:]:
        J = J.union(pq_set)
    J = list(J)

    print(f'vector dimension: {len(J)}')

    tensors = [
        pq_gram.pqgram_to_tensor(pqgram, J) for pqgram in tqdm(pqIndex, desc="[convert tensor]")
    ]
    
    indexes = torch.Tensor(range(num_trees))

    w_pq.cross_validation_5_fold(tensors, labels)

    w_pq.test()

    end = time.time()

    print(f'\ntime: {end-start} sec')
    

def main2():
    w_pq.test()
    

if __name__=="__main__":
    main2()

