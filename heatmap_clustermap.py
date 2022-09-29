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
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import matplotlib.pylab as plt
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

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
row_order = hc.dendrogram(row_linkage)
col_order = hc.dendrogram(col_linkage)
heat=np.log1p(heat)
g=sns.clustermap(heat, figsize=(20, 20), row_linkage=row_linkage, col_linkage=col_linkage, vmin=0, vmax=5, xticklabels = False, yticklabels = False)
    # yticklabels = [dict[r] for r in row_order['leaves']], xticklabels = [dict[5000 + c] for c in col_order['leaves']]) 
g.ax_cbar.tick_params(labelsize=25)
ax = g.ax_heatmap
# ax.set_ylabel("cs_keywords")
# ax.set_xlabel("medical_keywords")

ax.set_yticklabels(ax.get_ymajorticklabels(), fontdict={'fontsize':7})
ax.set_xticklabels(ax.get_xmajorticklabels(), fontdict={'fontsize':7})

g.savefig("picture/clusterheat_2022_no_axis.png")


