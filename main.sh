#!/bin/bash

# 1つ目のスクリプトを実行
python corpus_vs_corpus.py > logs/w-pq_unlabel/EWT_spacy_20.log

# 1つ目のスクリプトが終了したら2つ目のスクリプトを実行
python pq_knn_unlabel.py > logs/1nn-22gram_unlabel/EWT_spacy_20.log

code logs/w-pq_unlabel/EWT_spacy_20.log

code logs/1nn-22gram_unlabel/EWT_spacy_20.log