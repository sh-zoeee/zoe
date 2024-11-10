#!/bin/bash

targetCorpus="Japanese"
labelType="upos"


echo start: En vs ${targetCorpus}

python corpus_vs_corpus.py -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

python pq_knn_unlabel.py -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

##########
targetCorpus="French"

echo start: En vs ${targetCorpus}

python corpus_vs_corpus.py -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

python pq_knn_unlabel.py -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

##########
targetCorpus="Chinese"

echo start: En vs ${targetCorpus}

python corpus_vs_corpus.py -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

python pq_knn_unlabel.py -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log


##########
targetCorpus="German"

echo start: En vs ${targetCorpus}

python corpus_vs_corpus.py -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

python pq_knn_unlabel.py -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log


##########
targetCorpus="Korean"

echo start: En vs ${targetCorpus}

python corpus_vs_corpus.py -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

python pq_knn_unlabel.py -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log


##########
targetCorpus="Russian"

echo start: En vs ${targetCorpus}

python corpus_vs_corpus.py -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

python pq_knn_unlabel.py -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

##########
targetCorpus="Spanish"

echo start: En vs ${targetCorpus}

python corpus_vs_corpus.py -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

python pq_knn_unlabel.py -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log


##########
targetCorpus="English"

echo start: En vs ${targetCorpus}

python corpus_vs_corpus.py -t ${targetCorpus} > logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/w-pq_${labelType}/PUD_${targetCorpus}_50.log

python pq_knn_unlabel.py -t ${targetCorpus} > logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log

echo saved to logs/1nn-22gram_${labelType}/PUD_${targetCorpus}_50.log
