import pandas as pd
import numpy as np
import gensim
import pickle
from gensim.models import Word2Vec
import pandas as pd
import nltk
from sklearn.linear_model import Lasso
import re
import glob
from gensim.models.phrases import Phrases, Phraser
import ahocorasick
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
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
with open("cs1000_combine.txt") as f:
    for line in f:
        if(line.find('_')!=-1):
            pos=line.strip().find(':')
            conbine=re.sub(r'_',' ',line.strip().lower()[pos+1:])
            if(conbine!=last_conbine):
                tot_cs+=1
            last_conbine=conbine
            C.add_word(re.sub(r'_',' ',line.strip().lower()[:pos]),tot_cs)
            dict[tot_cs]=conbine
        

with open("medical1000_combine.txt") as f:
    for line in f:
        if(line.find('_')!=-1):
            pos=line.strip().find(':')
            conbine=re.sub(r'_',' ',line.strip().lower()[pos+1:])
            if(conbine!=last_conbine):
                tot_medical+=1
            last_conbine=conbine
            M.add_word(re.sub(r'_',' ',line.strip().lower()[:pos]),tot_medical)
            dict[5000+tot_medical]=conbine


C.make_automaton()
M.make_automaton()

df_all = pd.read_csv('./papers.csv')
start=1
end=12
df_all['Date']=(df_all['year']-1970)*12+df_all['month']


last_heat=np.zeros((tot_cs+100,tot_medical+100))
last_cs=np.zeros(tot_cs+100)
last_medical=np.zeros(tot_medical+100)
while(end<=600):#234: 2019-12

    #for c in range(1,tot_cs+1):
    #    for m in range(1,tot_medical+1):
    #        print(last_heat[c+20][m+20]," ")
    #print("\n")
    df=df_all[ (df_all['Date']>=start) &(df_all['Date']<=end)]
    df = df.reset_index(drop=True)
    #display(df)
    maxn=0
    heat=np.zeros((tot_cs+100,tot_medical+100))
    cs=np.zeros(tot_cs+100)
    medical=np.zeros(tot_medical+100)
    for i in range(len(df)):
        #print(i)
        if(pd.isnull(df.loc[i]['abstract'])):
            continue
        medical_word=[]
        cs_word=[]
        for end_index, c in C.iter(df.loc[i]['abstract'].lower()):
            cs_word.append(c)
        for end_index, m in M.iter(df.loc[i]['abstract'].lower()):
            medical_word.append(m)
        for c in set(cs_word):
            cs[c]+=1
        for m in set(medical_word):
            medical[m]+=1
        for c in set(cs_word):
            for m in set(medical_word):
                heat[c,m]+=1
    print("___________________________________________________________")
    print("Result of Year  ",int(start/12)+1970,"  :")
    print(len(df))
    start+=12
    end+=12
    last_heat=heat
    last_cs=cs
    last_medical=medical

