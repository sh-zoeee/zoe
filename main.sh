#!/bin/bash

sourceCorpus="Ja-BCCWJ"
labelType="unlabel"


targetCorpus="Ko-Kaist"


echo start: ${sourceCorpus} vs ${targetCorpus}

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} > logs/w-pq_${labelType}/${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/${sourceCorpus}_${targetCorpus}_50.log

python pq_knn_unlabel.py -s ${sourceCorpus} -t ${targetCorpus} > logs/1nn-22gram_${labelType}/${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/${sourceCorpus}_${targetCorpus}_50.log
