import pyconll
import pyconll.util
import numpy as np
import re
import sys
import random

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import mpl_toolkits.axisartist as AA
import matplotlib.patches as mpatches

import scripts.py.original.CFG.genCF as genCF


class MyToken():
    def __init__(self,form,index,head):
        self.form = form
        self.index = index
        self.head = head
        
    def str(self):
        return str(self.form)+"(index="+str(self.index)+",head="+str(self.head)+")"

    def getForm(self):
        return self.form
    
    def getIndex(self):
        return self.index

    def setId(self,index):
        return self.index
    
    def setHead(self,head):
        self.head = head

    def getHead(self):
        return self.head

        

def printDepAll(dp):
   for id in dp.keys():
        print(str(id)+"->"+str(dp[id]))


def printTokens(sentence):

    print("-------------sent_id="+sentence.id+" length="+str(len(sentence)))
    for token in sentence:    
        print("--------")
        print("id=",token.id)
        print("head=",token.head)
        print("lemma=",token.lemma)
        print("upos=",token.upos)
        print("xpos=",token.xpos)
        print("feats=",token.feats)
        print("deprel=",token.deprel)
        print("deps=",token.deps)
        print("misc=",token.misc)


def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()


def getSentenceLen(sentence):

    ids = []

    for token in sentence:
        if is_integer(token.id):
            ids.append(int(token.id))
            
    return max(ids)


def joinGrammar(ga,gb,grammarBefore,grammarAfter,numBefore,numAfter):

    for token in ga.keys():
        word = token.form
        if word not in grammarAfter:
            grammarAfter[word] = []
        if word not in numAfter:
            numAfter[word] = []            
        numAfter[word].append(len(ga[token]))
        for val in ga[token]:
            grammarAfter[word].append(val.lemma)

    for token in gb.keys():
        word = token.form
        if word not in grammarBefore:
            grammarBefore[word] = []
        if word not in numBefore:
            numBefore[word] = []            
        numBefore[word].append(len(gb[token]))
        for val in gb[token]:
            grammarBefore[word].append(val.lemma)            
        


def buildGrammar(filename):
    
    sentences = pyconll.load_from_file(filename)
    sys.stderr.write("Build grammar from "+str(len(sentences))+" sentences.\n")
    
    grammarBefore = {}
    grammarAfter = {}    
    heads = []
    freq = {}
    lengths = []
    sentence_num = 0
    numBefore = {}
    numAfter = {}
    word_to_upos = {}
    
    for sentence in sentences:

        ln = getSentenceLen(sentence)
        if ln < 4:
            continue
        
        lengths.append(getSentenceLen(sentence))
        sentence_num = sentence_num + 1        

        #printTokens(sentence)

        id2word = {}

        for token in sentence:
            if token.lemma == None:
                continue
            if token.lemma not in freq:
                freq[token.lemma] = 0
            freq[token.lemma] = freq[token.lemma] + 1 
            id2word[token.id] = genCF.MyWord(token.lemma, None, upos=token.upos)
            word_to_upos[token.lemma] = token.upos

            #print(f"単語: {token.lemma}, 品詞: {token.upos}")
        #print(id2word)

        gb = {}
        ga = {}
        
        for token in sentence:
            if token.lemma == None:
                continue
            
            if token.head == None:
                continue

            if token.head == "0":
                heads.append(token.lemma)

            else:

                if token.head not in id2word:
                    continue
   
                
                if token.id < token.head:
                    if id2word[token.head] not in gb:
                        gb[id2word[token.head]] = []                    
                    gb[id2word[token.head]].append(token)
                else:
                    if id2word[token.head] not in ga:
                        ga[id2word[token.head]] = []
                    ga[id2word[token.head]].append(token)
                    
        joinGrammar(ga,gb,grammarBefore,grammarAfter,numBefore,numAfter)
        
    return grammarBefore, grammarAfter, numBefore,numAfter,heads,freq, lengths, word_to_upos


def main():

    filename = "tmp/English-PUD.conllu"
    #filename = "tmp/test.conllu"

    grammarBefore,grammarAfter,numBefore,numAfter,heads,freq,lengths, word_to_upos = buildGrammar(filename)
    


    """
    print(heads)
    print("grammarBefore")
    print(grammarBefore)
    print("numBefore")
    print(numBefore)
    print("grammarAfter")
    print(grammarAfter)
    print("numAfter")
    print(numAfter)
    """
    print(word_to_upos)
    

if __name__== "__main__":
    main()

