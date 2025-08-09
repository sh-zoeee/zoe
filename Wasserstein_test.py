import pyconll
import pyconll.util
import numpy as np
#from scipy.optimize import curve_fit
import re
import sys
import datetime

from os import listdir
import os

import pandas as pd
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.manifold import TSNE

import random

from tqdm import tqdm  # forループの進捗バーを表示するライブラリ

from zss import simple_distance
import tree

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import mpl_toolkits.axisartist as AA
import matplotlib.patches as mpatches

from bifurcationRatio import calcRatio
import mydefs
import miscs

import ot

import matplotlib.cm as cm

import warnings
#from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
#warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
#warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.filterwarnings('ignore')

import umap

import wtestDist


from multiprocessing import Pool
import multiprocessing

def calcDistOne(treepair):
    return float(simple_distance(treepair[0], treepair[1]))


def calcDistPar(setTreeN):

    numData = len(setTreeN)
    cpu_num = multiprocessing.cpu_count()
    print(".....calculating distances among "+str(numData)+"trees with",cpu_num, "cores.")
    
    distArray = np.zeros((numData, numData), dtype=float)
    
    pairs = []
    for i in range(numData):
        for j in range(i):
            pairs.append((setTreeN[i],setTreeN[j]))

    p = Pool(cpu_num) # プロセス数を4に設定            
    #result = p.map(calcDistOne, tqdm(pairs))
    result = p.map(calcDistOne, tqdm(pairs))
    #print("len=",len(result))
    
    count = 0
    for i in range(numData):
        distArray[i][i] = 0
        for j in range(i):
            distArray[i][j] = result[count]
            distArray[j][i] = result[count]
            count += 1

            
    return distArray
                


def calcDist(sentTreeN):
             
    numData = len(sentTreeN)
    print(".....calculating distances among "+str(numData)+" trees")
    
    distArray = np.zeros((numData, numData), dtype=float)
    
    for i in tqdm(range(numData)):
        distArray[i][i] = 0
        for j in range(i):
            dis = float(simple_distance(sentTreeN[i],sentTreeN[j]))
            distArray[i][j] = dis
            distArray[j][i] = dis

    return distArray

def getLabel(name):
    prg = re.compile('.+/(.+)\.conllu')
    result = prg.match(name)
    #print(result.group(1))
    return result.group(1)


def bval(D, r, s):
    n = D.shape[0]
    total_r = np.sum(D[:,s] ** 2)
    total_s = np.sum(D[r,:] ** 2)
    total = np.sum(D ** 2)
    val = (D[r,s] ** 2) - (float(total_r) / float(n)) - (float(total_s) / float(n)) + (float(total) / float(n * n))
    return -0.5 * val


def getLang(corpusname):

    langprefix = re.compile('([^\-]+)-.+')
    result = langprefix.match(corpusname)
    if not result:
        print("Filename not correct!")
       
    return result.group(1)
    


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
            
    if len(ids)==0:
        return 0
            
    return max(ids)


def getLangFromNum(k,nonZeroFiles,nums):

    s = 0
    for t in range(len(nums)):
        s += nums[t]
        if k < s:
            break
    return nonZeroFiles[t]


def checkDuplicate(samples):
    aaa = [ ]
    for sentence in samples:
        sentstr = ""
        for t in sentence:            
            sentstr += str(t.id)+"-"+str(t.head)+"="
        aaa.append(sentstr)

    print("set",len(set(aaa)))
    print("list",len(aaa))
    exit(0)



def getCums(nums):

    total = 0
    cums = [total]
    for num in nums:
        total += num
        cums.append(total)

    return cums


def filterSentence(filterflag,lang,sentence,LEN_SENTENCE):

    if filterflag:
        return getSentenceLen(sentence) == LEN_SENTENCE        

    SIGMA_ALL = 0.728
    MU_ALL = 1.456

    sentlen = getSentenceLen(sentence)
    
    if sentlen == LEN_SENTENCE:
        if lang == "Rand_English-EWT":
            branchnumall,leafnumall = calcRatio(sentence.to_tree())
            val = float(branchnumall)/float(sentlen-leafnumall)
            return val > MU_ALL-SIGMA_ALL and val < SIGMA_ALL + SIGMA_ALL
        else:
            return True
    
    return False


def readFilterSentences(fs,LEN_SENTENCE,THRESHOLD_MIN_NUM):

    sentences = {}
    DIRTMP = "data/original/treebank/English/"
    SUFFIX = ".conllu"
    for filename in fs:
        print(DIRTMP + filename)
        sentences[filename] = pyconll.load_from_file(DIRTMP + filename + SUFFIX)

        numSentences= len(sentences[filename])
        print("numSentences="+str(numSentences))

        ls = []   # put all sentences of length LEN_SENTENCE
        for sent in sentences[filename]:
            #if filterSentence(filterflag,filename,sent,LEN_SENTENCE):
            if getSentenceLen(sent) == LEN_SENTENCE:
                ls.append(sent)
                
        if len(ls) == THRESHOLD_MIN_NUM:
            continue

        print("found",str(len(ls))," sentences of length ",str(LEN_SENTENCE))

        sentences[filename] = ls
        
    return sentences        

def calculateDistanceMatrix(fs,MAX_SAMPLE_NUM,dist_kind,sentences_lenfiltered):

    nums = []
    sentTreeN = []
    targetSentences = []
    nonZeroFiles = []
    
    DIRTMP = "data/original/treebank/English"
    SUFFIX = ".conllu"
    
    for filename in fs:

        ls = sentences_lenfiltered[filename]
        samples = ls
        if len(ls) > MAX_SAMPLE_NUM:
            samples = random.sample(ls,MAX_SAMPLE_NUM) 
            nums.append(MAX_SAMPLE_NUM)
        else:
            nums.append(len(ls))
        nonZeroFiles.append(filename)
       
        for sent in samples:
            targetSentences.append(sent)
            ud_tree = sent.to_tree()            
            mytree = tree.mkTree(ud_tree,dist_kind)
            #print(mytree)
            sentTreeN.append(mytree)
        
    return calcDistPar(sentTreeN),nums,nonZeroFiles,targetSentences,sentTreeN



def testOneTime(fs,MAX_SAMPLE_NUM,dist_kind,sentences,BootStrapNum,Snum):

    
    distArray,nums,nonZeroFiles,targetSentences,sentTreeN = calculateDistanceMatrix(fs,MAX_SAMPLE_NUM,dist_kind,sentences)

    baseClass = "English-EWT"
    counterClass = nonZeroFiles[1]
        
    try:
        class_base = nonZeroFiles.index(baseClass)
        class_counter = nonZeroFiles.index(counterClass)
    except ValueError:
        print("No class of",counterClass, "in", nonZeroFiles )   
        exit(0)
            
    #rank_vals.append(wtestDist.doOneTest(distArray,class_base,class_counter,nums,nonZeroFiles,BootStrapNum))
    rank,wmn_val_s = wtestDist.doOneTestPar(distArray,class_base,class_counter,nums,nonZeroFiles,BootStrapNum)
        
    pvalue = rank / float(BootStrapNum)

    print("pvalue  = ",pvalue)
    print("wasserstein distance  = ",wmn_val_s)

    return pvalue,wmn_val_s


def main():


    tedDists =  ["NonOrd","Ord","Dir","DistFromHead"]   # must be defined in tree.py
    #setDists =  ["Hausdorf","meanEuclid","meanTED","Wasserstein"]   
           # must be defined in the function calcDendrogram
    
    if len(sys.argv) < 5:
        print("Format: python distTtest.py languageSet distKind LenSentence MaxNumSamples opt:testClass")
        print("\tFor example: english Ord N=20 K=20:  for Test, K must be small")
        print("\tLanguage set must be defined in mydefs.py")
        print("\tDist kind must be one of "+" ".join(tedDists))
   
        print("\t   -NonOrd: unlabeled ordered tree")
        print("\t   -Ord: orered tree with label being the offset of the word in the sentence")
        print("\t   -Dir: labeled ordered tree; label is before(-1)/after(+1), sentence head 0")
        print("\t   -DistFromHead: labeled ordered tree; label is the signed distance from the head")
        print("\tOptional: testClass is \"all\", or one class in the language set.")
        exit(0)

    lang_set = sys.argv[1]
    dist_kind = sys.argv[2]
    ln_sntnc = sys.argv[3]
    LEN_SENTENCE = int(ln_sntnc)
    mx_smpl_nm = sys.argv[4]
    MAX_SAMPLE_NUM = int(sys.argv[4])

    testClass = "is the 2nd file"

    #print(sys.argv)
    if len(sys.argv) == 6:
        testClass = sys.argv[5]
    
    print("language_set = "+lang_set)
    print("dist_kind = "+dist_kind)
    print("LEN_SENTENCE = "+ln_sntnc)
    print("MAX_SAMPLE_NUM = "+mx_smpl_nm)
    print("testClass=",testClass)                
    if dist_kind not in tedDists:
        print("Distance kind not defined")
        exit(0)

    try:
        fs = eval("mydefs."+lang_set)            
    except AttributeError:
        print("In mydefs.py, no set of languages defined: "+lang_set)
        exit(0)

    print(fs)
        
    #outfilename = lang_set+"_"+dist_kind+"_"+ln_sntnc+"_"+mx_smpl_nm
    #print(outfilename)

    ####################

    Snum = 1
    
    THRESHOLD_MIN_NUM = 10

    xs = []
    ys = []
    ystd = []

    
    repeat_times = 10
    #repeat_times = 3

    BootStrapNum = 1
    
    sentence_lens = 20
    
    sentences = readFilterSentences(fs,sentence_lens,THRESHOLD_MIN_NUM)

    vals = []
    disar = []
    for k in range(repeat_times):
    
        pvalue,dis = testOneTime(fs,MAX_SAMPLE_NUM,dist_kind,sentences,BootStrapNum,Snum)
        vals.append(pvalue)
        disar.append(dis)


    dir = "wtest_stats/"
    fp = open(dir+str(MAX_SAMPLE_NUM)+"_mean_vals"+"_"+lang_set+"_"+dist_kind+".dat","w")

    result = "%-5.3f$\pm$%-5.3f" %  (np.mean(disar),np.std(disar))
    if BootStrapNum == 1:
        print(result,file=fp)
    fp.close()

    if BootStrapNum > 1:
        print("pvalue mean=",np.mean(vals))
        print("pvalue std=",np.std(vals))
        
    print("dis mean=",np.mean(disar))
    print("dis std=",np.std(disar))    
    
    #fp.close()


if __name__== "__main__":
    main()


