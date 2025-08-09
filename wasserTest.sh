#!/bin/bash

ewt="English-EWT"



#python scripts/py/wasserTest.py ${ewt} English-EWT
python scripts/py/wasserTest.py ${ewt} English-ESL
python scripts/py/wasserTest.py ${ewt} English-Atis

python scripts/py/wasserTest.py ${ewt} English-RandCF
python scripts/py/wasserTest.py ${ewt} English-ChatGPT

python scripts/py/wasserTest.py ${ewt} English-EWT-spacy
python scripts/py/wasserTest.py ${ewt} English-EWT-udpipe

python scripts/py/wasserTest.py ${ewt} Rand-Uniform
python scripts/py/wasserTest.py ${ewt} Rand-Markov2
python scripts/py/wasserTest.py ${ewt} Rand-Markov3
python scripts/py/wasserTest.py ${ewt} Rand-Markov5
python scripts/py/wasserTest.py ${ewt} Rand-Markov10

python scripts/py/wasserTest.py ${ewt} Arabic-NYUAD
python scripts/py/wasserTest.py ${ewt} Belarusian-HSE
python scripts/py/wasserTest.py ${ewt} Catalan-AnCora
python scripts/py/wasserTest.py ${ewt} Chinese-GSD
python scripts/py/wasserTest.py ${ewt} Czech-PDT
python scripts/py/wasserTest.py ${ewt} Estonian-EDT
python scripts/py/wasserTest.py ${ewt} French-FTB
python scripts/py/wasserTest.py ${ewt} German-HDT
python scripts/py/wasserTest.py ${ewt} Icelandic-IcePaHC
python scripts/py/wasserTest.py ${ewt} Italian-ISDT
python scripts/py/wasserTest.py ${ewt} Latvian-LVTB
python scripts/py/wasserTest.py ${ewt} Norwegian-Bokmaal
python scripts/py/wasserTest.py ${ewt} Persian-PerDT
python scripts/py/wasserTest.py ${ewt} Polish-PerDT
python scripts/py/wasserTest.py ${ewt} Portuguese-GSD
python scripts/py/wasserTest.py ${ewt} Romanian-Nonstandard
python scripts/py/wasserTest.py ${ewt} Russian-SynTag
python scripts/py/wasserTest.py ${ewt} Spanish-Ancora


