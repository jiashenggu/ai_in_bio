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
        pos=line.strip().find(':')
        conbine=re.sub(r'_',' ',line.strip().lower()[pos+1:])
        if(conbine!=last_conbine):
            tot_cs+=1
        last_conbine=conbine
        C.add_word(re.sub(r'_',' ',line.strip().lower()[:pos]),tot_cs)
        dict[tot_cs]=conbine
        

with open("medical1000_combine.txt") as f:
    for line in f:
        pos=line.strip().find(':')
        conbine=re.sub(r'_',' ',line.strip().lower()[pos+1:])
        if(conbine!=last_conbine):
            tot_medical+=1
        last_conbine=conbine
        M.add_word(re.sub(r'_',' ',line.strip().lower()[:pos]),tot_medical)
        dict[5000+tot_medical]=conbine


C.make_automaton()
M.make_automaton()

# df_all = pd.read_csv('./papers.csv')
df_all = pd.read_csv('./papers_2022.csv')
start=1+60*9
end=60+60*9
#  1970
df_all['Date']=(df_all['year']-1970)*12+df_all['month']

print(tot_cs,"  ",tot_medical)
last_heat=np.zeros((tot_cs+100,tot_medical+100))
train_data={}
train_label={}
test_data={}
test_label={}
for c in range(1,tot_cs+1):
    for m in range(1,tot_medical+1):
        train_data[c*10000+m]=[]
        train_label[c*10000+m]=[]
        test_data[c*10000+m]=[]
        test_label[c*10000+m]=[]
# while(end<=600):#234: 2019-12
while(end <= 624): # 2021-12
    print(start,"   ",end, len(train_label))
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
    print("TOP20 CS:")
    tmp=[]
    name=[]
    for c in range(0,tot_cs+1):
        tmp.append(cs[c])
        name.append(dict[c])
    tmp,name=zip(*sorted(zip(tmp, name)))
    names = []
    tmps = []
    total = np.sum(cs)
    for i in range(20):
        names.append(name[len(name)-1-i])
        tmps.append(tmp[len(name)-1-i]/total*100)
        # print(name[len(name)-1-i],":",tmp[len(name)-1-i])
    df = pd.DataFrame({'name': names, '%': tmps})
    df.index = range(1, 21)
    print(df.style.to_latex())
    print(" ")
    print("TOP20 medical:")
    tmp=[]
    name=[]
    for m in range(0,tot_medical+1):
        tmp.append(medical[m])
        name.append(dict[m+5000])
    tmp,name=zip(*sorted(zip(tmp, name)))
    names = []
    tmps = []
    total = np.sum(medical)
    for i in range(20):
        names.append(name[len(name)-1-i])
        tmps.append(tmp[len(name)-1-i]/total*100)
        # print(name[len(name)-1-i],":",tmp[len(name)-1-i])
    df = pd.DataFrame({'name': names, '%': tmps})
    df.index = range(1, 21)
    print(df.style.to_latex())
    print(" ")  
    print("TOP20 cs&medical:")
    tmp=[]
    name=[]
    for c in range(0,tot_cs+1):
        for m in range(0,tot_medical+1):
            tmp.append(heat[c,m])
            name.append(dict[c]+" , "+dict[m+5000])
    tmp,name=zip(*sorted(zip(tmp, name)))
    names = []
    tmps = []
    total = np.sum(heat)
    for i in range(20):
        names.append(name[len(name)-1-i])
        tmps.append(round(tmp[len(name)-1-i]/total*100, 3))
        # print(name[len(name)-1-i],":",tmp[len(name)-1-i])
    df = pd.DataFrame({'name': names, '%': tmps})
    df.index = range(1, 21)
    print(df.to_latex(float_format="%.3f"))
    start+=60
    end+=60


