#!/bin/bash

sourceCorpus="Japanese"
labelType="upos"


targetCorpus="Japanese"


echo start: ${sourceCorpus} vs ${targetCorpus}

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

python pq_knn_unlabel.py -s ${sourceCorpus} -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

##########
targetCorpus="French"

echo start: ${sourceCorpus} vs ${targetCorpus}

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

python pq_knn_unlabel.py -s ${sourceCorpus} -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

##########
targetCorpus="Chinese"

echo start: ${sourceCorpus} vs ${targetCorpus}

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

python pq_knn_unlabel.py -s ${sourceCorpus} -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log


##########
targetCorpus="German"

echo start: ${sourceCorpus} vs ${targetCorpus}

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

python pq_knn_unlabel.py -s ${sourceCorpus} -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log


##########
targetCorpus="Korean"

echo start: ${sourceCorpus} vs ${targetCorpus}

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

python pq_knn_unlabel.py -s ${sourceCorpus} -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log


##########
targetCorpus="Russian"

echo start: ${sourceCorpus} vs ${targetCorpus}

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

python pq_knn_unlabel.py -s ${sourceCorpus} -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

##########
targetCorpus="Spanish"

echo start: ${sourceCorpus} vs ${targetCorpus}

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

python pq_knn_unlabel.py -s ${sourceCorpus} -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log


##########
targetCorpus="English"

echo start: ${sourceCorpus} vs ${targetCorpus}

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

python pq_knn_unlabel.py -s ${sourceCorpus} -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${sourceCorpus}_${targetCorpus}_50.log
