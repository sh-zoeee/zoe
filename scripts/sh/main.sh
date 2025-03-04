#!/bin/bash

sourceCorpus="English-EWT"
labelType="unlabel"


targetCorpus="French-GSD"


echo start: ${sourceCorpus} vs ${targetCorpus} ${labelType}

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} > logs/w-pq_${labelType}/${sourceCorpus}_${targetCorpus}_20.log

echo saved to logs/w-pq_${labelType}/${sourceCorpus}_${targetCorpus}_20.log

python pq_knn_unlabel.py -s ${sourceCorpus} -t ${targetCorpus} > logs/1nn-22gram_${labelType}/${sourceCorpus}_${targetCorpus}_20.log

echo saved to logs/1nn-22gram_${labelType}/${sourceCorpus}_${targetCorpus}_20.log
