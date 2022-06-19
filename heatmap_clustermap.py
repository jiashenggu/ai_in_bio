import pandas as pd
import numpy as np
import gensim
import pickle
from gensim.models import Word2Vec
import pandas as pd
import nltk
import re
import glob
from gensim.models.phrases import Phrases, Phraser
import ahocorasick
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
C = ahocorasick.Automaton()
M = ahocorasick.Automaton()
def not_all_uppercase(str):
    for i in range(len(str)):
        if(str[i]>='a' and str[i]<='z'):
            return True
    return False

tot_cs=-1
tot_medical=-1
dict={}
last_conbine="hahhahahahahahahaha"
with open("top200cs_combine.txt") as f:
    for line in f:
        pos=line.strip().find(':')
        conbine=re.sub(r'_',' ',line.strip().lower()[pos+1:])
        if(conbine!=last_conbine):
            tot_cs+=1
        last_conbine=conbine
        C.add_word(re.sub(r'_',' ',line.strip().lower()[:pos]),tot_cs)
        dict[tot_cs]=line.strip().lower()[:pos]
        print(line.strip().lower()[:pos],"    ",conbine,"    ",tot_cs)

with open("top200medical_combine.txt") as f:
    for line in f:
        pos=line.strip().find(':')
        conbine=re.sub(r'_',' ',line.strip().lower()[pos+1:])
        if(conbine!=last_conbine):
            tot_medical+=1
        last_conbine=conbine
        M.add_word(re.sub(r'_',' ',line.strip().lower()[:pos]),tot_medical)
        dict[5000+tot_medical]=line.strip().lower()[:pos]

C.make_automaton()
M.make_automaton()
# model = Word2Vec.load("word2vec/fulltext_abstract_phrases3.model")
model = Word2Vec.load("word2vec/fulltext_abstract_new.model")
# df = pd.read_csv('./papers.csv')
df = pd.read_csv('./papers_2022.csv')

print(tot_cs,"  ",tot_medical)
heat=np.zeros((tot_cs+1,tot_medical+1))

for i in range(len(df)):
    #print(i)
    if(pd.isnull(df.loc[i]['abstract'])):
        continue
    str=df.loc[i]['abstract']
    medical_word=[]
    cs_word=[]
    for end_index, c in C.iter(df.loc[i]['abstract'].lower()):
        cs_word.append(c)
    for end_index, m in M.iter(df.loc[i]['abstract'].lower()):
        medical_word.append(m)
    for c in set(cs_word):
        for m in set(medical_word):
            heat[c,m]+=1
            
import seaborn as sns;
import scipy.spatial as sp, scipy.cluster.hierarchy as hc

row_dis=[]
col_dis=[]
for i in range(0,tot_cs+1):
    for j in range(i+1,tot_cs+1):
        row_dis.append(1-model.wv.similarity(dict[i],dict[j]))
        
for i in range(0,tot_medical+1):
    for j in range(i+1,tot_medical+1):
        col_dis.append(1-model.wv.similarity(dict[5000+i],dict[5000+j]))
row_linkage = hc.linkage(row_dis, method='average')
col_linkage= hc.linkage(col_dis, method='average')
heat=np.log1p(heat)
g=sns.clustermap(heat, row_linkage=row_linkage, col_linkage=col_linkage ) 
g.savefig("picture/clusterheat_2022.pdf")


