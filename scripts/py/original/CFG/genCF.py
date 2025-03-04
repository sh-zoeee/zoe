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

from tqdm import tqdm 

import scripts.py.original.CFG.buildGrammar as buildGrammar 

class MyWord():
    def __init__(self,form,head, upos):
        self.form = form
        self.head = head   #MyWord型
        self.upos = upos
        
    def getForm(self):
        return self.form
    
    def getIndex(self):
        return self.index
    
    def getUpos(self):
        return self.upos

    def setIndex(self,id):
        self.index = id    

    def setHead(self,head):
        self.head = head   #MyWord型

    def getHeadId(self):
        if self.head ==0:
            return 0
        return self.head.getIndex()


    
def generateOne(myWord,grammar,freq, children_nums, word_to_upos):
    
    
    result = []
    word = myWord.form
    upos = myWord.upos
    #print(word)

    if freq[word] == 0:
        return [myWord]

    if word not in grammar:
        return [myWord]

    num = random.sample(children_nums[word],1)[0]
    #print("num=",num," children for", word)

    before = []
    after = []
    
    for i in range(num):
        
        sample_word = random.sample(grammar[word],1)[0]
        position = random.choice([-1,1])
        #print(position)

        #print("position=",position," sample_word", sample_word)

        newChild = MyWord(sample_word,myWord,upos=word_to_upos[sample_word])

        if position==-1:
            #print("a")
            before.append(newChild)
        elif position == 1:
            #print("b")
            after.append(newChild)

    #print(before, after)

    for ch in before:
        result_ch = generateOne(ch,grammar,freq,children_nums,word_to_upos)
        result += result_ch            
    result.append(myWord)
    for ch in after:
        result_ch = generateOne(ch,grammar,freq,children_nums,word_to_upos)
        result += result_ch
        
    return result

        

def generate(SENTENCE_NUM,grammar,heads, freq, lengths,children_nums, word_to_upos,fp):
    
    result_all = []
    sys.setrecursionlimit(10000)
    
    for i in tqdm(range(SENTENCE_NUM)):

        fp.write("# sent_id = "+str(i)+"\n")
            
        head = random.sample(heads,1)[0]
        head_upos = word_to_upos[head]
        #print(head, head_upos)
        index = 1
        head_word = MyWord(head,0,upos=head_upos)

        try:
            sentence = generateOne(head_word,grammar,freq,children_nums,word_to_upos)
        except RecursionError:
            i -= 1
            continue
        
        text = "" # string型のsentence
        for w in sentence:
            text += " " + w.getForm()
        
        fp.write("# text =" + text + "\n")
        
        id = 1
        for w in sentence:
            w.setIndex(id)
            id += 1
        
        for w in sentence:

            tuple = [str(w.getIndex()),w.getForm(),w.getForm(),w.getUpos(),"_","_",str(w.getHeadId()),"_","_","_"]
            fp.write("\t".join(tuple))
            fp.write("\n")
            #print("\t".join(tuple))            
                
        fp.write("\n")
        
    return result_all


def main():

    
    filename = "tmp/English-PUD.conllu"
    #filename = "tmp/test0.conllu"
    #filename = "tmp/test.conllu"    

    SENTENCE_NUM = 10
    
    if len(sys.argv) < 3:
        print("Format: python genCF.py SENTENCE_NUM outfile")
        print("\tExample: python genCF.py 100 outfile")
        print("outfile is gendata/outfile.conllu")
        exit(0)
        
    grammar,_, num_before,children_nums,heads,freq,lengths, word_to_upos = buildGrammar.buildGrammar(filename)

    SENTENCE_NUM = int(sys.argv[1])

    fp = open(sys.argv[2], "w")    

    generate(SENTENCE_NUM,grammar,heads, freq, lengths,num_before, word_to_upos,fp)

if __name__== "__main__":
    main()
