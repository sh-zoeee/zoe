
#CORPORA_DIR = "/home/yamazoe/zoe/data/original/UD-2.9/"
from sqlite3 import PARSE_DECLTYPES


CORPORA_DIR = "/home/yamazoe/tree_data/UD_NL/"
RAND_DIR = "/home/yamazoe/tree_data/rand/"
PARSER_DIR = "/home/yamazoe/tree_data/parser/"

POSTFIX_CONLLU = ".conllu"

DISTMX_DIR = "/home/yamazoe/zoe/data/numpy_data/distmx/ewt_other/"

LOG_DIR = "/home/yamazoe/zoe/logs/iclr2025/"

CORPORA_PATH = {
    #### English ####
    "English-Atis": CORPORA_DIR+"English-Atis"+POSTFIX_CONLLU,
    "English-CHILDES": CORPORA_DIR+"English-CHILDES"+POSTFIX_CONLLU,
    "English-CTeTex": CORPORA_DIR+"English-CTeTex"+POSTFIX_CONLLU,
    "English-ESL": CORPORA_DIR+"English-ESL"+POSTFIX_CONLLU,
    "English-ESLSpok": CORPORA_DIR+"English-ESLSpok"+POSTFIX_CONLLU,
    "English-EWT": CORPORA_DIR+"English-EWT"+POSTFIX_CONLLU,
    "English-GENTLE": CORPORA_DIR+"English-GENTLE"+POSTFIX_CONLLU,
    "English-GUM": CORPORA_DIR+"English-GUM"+POSTFIX_CONLLU,
    "English-GUMReddit": CORPORA_DIR+"English-GUMReddit"+POSTFIX_CONLLU,
    "English-LinES": CORPORA_DIR+"English-LinES"+POSTFIX_CONLLU,
    "English-ParTUT": CORPORA_DIR+"English-ParTUT"+POSTFIX_CONLLU,
    "English-Pronouns": CORPORA_DIR+"English-Pronouns"+POSTFIX_CONLLU,
    "English-PUD": CORPORA_DIR+"English-PUD"+POSTFIX_CONLLU,

    #### Rand ####
    "English-EWT-RandCF": RAND_DIR+"English-EWT_RandCF"+POSTFIX_CONLLU,

    "Rand-GPT4o-spacy": RAND_DIR+"Rand-GPT4o-spacy"+POSTFIX_CONLLU,
    "Rand-GPT4o-udpipe": RAND_DIR+"Rand-GPT4o-udpipe"+POSTFIX_CONLLU,

    "Rand-Balanced": RAND_DIR+"Rand-Balanced"+POSTFIX_CONLLU,
    "Rand-Star": RAND_DIR+"Rand-Star"+POSTFIX_CONLLU,

    "Rand-Uniform": RAND_DIR+"Rand-Uniform"+POSTFIX_CONLLU,
    "Rand-Markov2": RAND_DIR+"Rand-Markov2"+POSTFIX_CONLLU,
    "Rand-Markov3": RAND_DIR+"Rand-Markov3"+POSTFIX_CONLLU,
    "Rand-Markov5": RAND_DIR+"Rand-Markov5"+POSTFIX_CONLLU,
    "Rand-Markov10": RAND_DIR+"Rand-Markov10"+POSTFIX_CONLLU,

    #### Parser ####
    "English-EWT-spacy": PARSER_DIR+"English-EWT-ReparsedSpacy"+POSTFIX_CONLLU,
    "English-EWT-udpipe": PARSER_DIR+"English-EWT-ReparsedUDPipe"+POSTFIX_CONLLU,
    
    #### Other NLs ####
    #"Arabic-NYUAD": CORPORA_DIR+"Arabic-NYUAD"+POSTFIX_CONLLU,
    "Arabic-PADT": CORPORA_DIR+"Arabic-PADT"+POSTFIX_CONLLU,
    "Belarusian-HSE": CORPORA_DIR+"Belarusian-HSE"+POSTFIX_CONLLU,
    "Bulgarian-BTB": CORPORA_DIR+"Bulgarian-BTB"+POSTFIX_CONLLU,
    "Catalan-AnCora": CORPORA_DIR+"Catalan-AnCora"+POSTFIX_CONLLU,
    "Chinese-GSD": CORPORA_DIR+"Chinese-GSD"+POSTFIX_CONLLU,
    "Croatian-SET": CORPORA_DIR+"Croatian-SET"+POSTFIX_CONLLU,
    "Czech-PDT": CORPORA_DIR+"Czech-PDT"+POSTFIX_CONLLU,
    "Dutch-LassySmall": CORPORA_DIR+"Dutch-LassySmall"+POSTFIX_CONLLU,
    "Estonian-EDT": CORPORA_DIR+"Estonian-EDT"+POSTFIX_CONLLU,
    "Finnish-FTB": CORPORA_DIR+"Finnish-FTB"+POSTFIX_CONLLU,
    "French-GSD": CORPORA_DIR+"French-GSD"+POSTFIX_CONLLU,
    "German-HDT": CORPORA_DIR+"German-HDT"+POSTFIX_CONLLU,
    "Hindi-HDTB": CORPORA_DIR+"Hindi-HDTB"+POSTFIX_CONLLU,
    "Icelandic-IcePaHC": CORPORA_DIR+"Icelandic-IcePaHC"+POSTFIX_CONLLU,
    "Italian-ISDT": CORPORA_DIR+"Italian-ISDT"+POSTFIX_CONLLU,
    #"Japanese-BCCWJ": CORPORA_DIR+"Japanese-BCCWJ"+POSTFIX_CONLLU,
    "Japanese-GSD": CORPORA_DIR+"Japanese-GSD"+POSTFIX_CONLLU,
    "Korean-Kaist": CORPORA_DIR+"Korean-Kaist"+POSTFIX_CONLLU,
    "Latvian-LVTB": CORPORA_DIR+"Latvian-LVTB"+POSTFIX_CONLLU,
    "Naija-NSC": CORPORA_DIR+"Naija-NSC"+POSTFIX_CONLLU,
    "Norwegian-Bokmaal": CORPORA_DIR+"Norwegian-Bokmaal"+POSTFIX_CONLLU,
    "Persian-PerDT": CORPORA_DIR+"Persian-PerDT"+POSTFIX_CONLLU,
    "Polish-PDB": CORPORA_DIR+"Polish-PDB"+POSTFIX_CONLLU,
    "Portuguese-CINTIL": CORPORA_DIR+"Portuguese-CINTIL"+POSTFIX_CONLLU,
    "Romanian-Nonstandard": CORPORA_DIR+"Romanian-Nonstandard"+POSTFIX_CONLLU,
    "Russian-SynTag": CORPORA_DIR+"Russian-SynTagRus"+POSTFIX_CONLLU,
    "Slovenian-SSJ": CORPORA_DIR+"Slovenian-SSJ"+POSTFIX_CONLLU,
    "Spanish-Ancora": CORPORA_DIR+"Spanish-AnCora"+POSTFIX_CONLLU,
    "Turkish-Kenet": CORPORA_DIR+"Turkish-Kenet"+POSTFIX_CONLLU,
    "Urdu-UDTB": CORPORA_DIR+"Urdu-UDTB"+POSTFIX_CONLLU,
}

PATH_ENGLISH_EWT = CORPORA_DIR+"English-EWT"+POSTFIX_CONLLU
PATH_JAPANESE_BCCWJ = CORPORA_DIR+"Japanese-BCCWJ"+POSTFIX_CONLLU
PATH_CHINESE_GSD = CORPORA_DIR+"Chinese-GSD"+POSTFIX_CONLLU
PATH_KOREAN_Kaist = CORPORA_DIR+"Korean-Kaist"+POSTFIX_CONLLU
PATH_FRENCH_GSD = CORPORA_DIR+"French-GSD"+POSTFIX_CONLLU

PLOT_COLORS = {
    #"Arabic-NYUAD": CORPORA_DIR+"Arabic-NYUAD"+POSTFIX_CONLLU,
    #"Belarusian-HSE": CORPORA_DIR+"Belarusian-HSE"+POSTFIX_CONLLU,
    #"Catalan-AnCora": CORPORA_DIR+"Catalan-AnCora"+POSTFIX_CONLLU,
    #"Chinese-GSD": CORPORA_DIR+"Chinese-GSD"+POSTFIX_CONLLU,
    #"Czech-PDT": CORPORA_DIR+"Czech-PDT"+POSTFIX_CONLLU,

    "English-ESL": "green",
    "English-Atis": "blue",
    "English-EWT": "red",

    "Rand-English-EWT-CF": "orange",
    "Rand-ChatGPT": "black",

    "Rand-Uniform": "cyan",
    "Rand-Markov2": "darkblue",
    "Rand-Markov3": "blue",
    "Rand-Markov5": "lightblue",
    "Rand-Markov10": "lightgray",

    "English-EWT-spacy": "green",
    "English-EWT-udpipe": "blue",
    
    #"Estonian-EDT": CORPORA_DIR+"Estonian-EDT"+POSTFIX_CONLLU,
    "French-GSD": "blue",
    #"French-FTB": CORPORA_DIR+"French-FTB"+POSTFIX_CONLLU,
    #"German-HDT": CORPORA_DIR+"German-HDT"+POSTFIX_CONLLU,
    #"Hindi-HDTB": CORPORA_DIR+"Hindi-HDTB"+POSTFIX_CONLLU,
    #"Icelandic-IcePaHC": CORPORA_DIR+"Icelandic-IcePaHC"+POSTFIX_CONLLU,
    #"Italian-ISDT": CORPORA_DIR+"Italian-ISDT"+POSTFIX_CONLLU,
    "Japanese-BCCWJ": "green",
    "Korean-Kaist": "lightgreen",
    #"Latvian-LVTB": CORPORA_DIR+"Latvian-LVTB"+POSTFIX_CONLLU,
    #"Norwegian-Bokmaal": CORPORA_DIR+"Norwegian-Bokmaal"+POSTFIX_CONLLU,
    #"Persian-PerDT": CORPORA_DIR+"Persian-PerDT"+POSTFIX_CONLLU,
    #"Polish-PDB": CORPORA_DIR+"Polish-PDB"+POSTFIX_CONLLU,
    #"Portuguese-GSD": CORPORA_DIR+"Portuguese-GSD"+POSTFIX_CONLLU,
    #"Romanian-Nonstandard": CORPORA_DIR+"Romanian-Nonstandard"+POSTFIX_CONLLU,
    #"Russian-SynTag": CORPORA_DIR+"Russian-SynTagRus"+POSTFIX_CONLLU,
    #"Spanish-Ancora": CORPORA_DIR+"Spanish-Ancora"+POSTFIX_CONLLU,
}

