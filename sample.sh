#!/bin/bash

sourceCorpus="English-EWT"
labelType="upos"


targetCorpus="English-EWT"

python corpus_vs_corpus.py -s ${sourceCorpus} -t ${targetCorpus} >> logs/w-pq_${labelType}/${sourceCorpus}_${targetCorpus}_50.log