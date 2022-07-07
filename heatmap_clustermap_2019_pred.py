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
import seaborn as sns
import scipy.spatial as sp
import scipy.cluster.hierarchy as hc
C = ahocorasick.Automaton()
M = ahocorasick.Automaton()


def not_all_uppercase(str):
    for i in range(len(str)):
        if(str[i] >= 'a' and str[i] <= 'z'):
            return True
    return False


tot_cs = -1
tot_medical = -1
dict = {}
last_conbine = "hahhahahahahahahaha"
with open("top200cs_combine.txt") as f:
    for line in f:
        pos = line.strip().find(':')
        conbine = re.sub(r'_', ' ', line.strip().lower()[pos+1:])
        if(conbine != last_conbine):
            tot_cs += 1
        last_conbine = conbine
        C.add_word(re.sub(r'_', ' ', line.strip().lower()[:pos]), tot_cs)
        dict[tot_cs] = line.strip().lower()[:pos]
        print(line.strip().lower()[:pos], "    ", conbine, "    ", tot_cs)

with open("top200medical_combine.txt") as f:
    for line in f:
        pos = line.strip().find(':')
        conbine = re.sub(r'_', ' ', line.strip().lower()[pos+1:])
        if(conbine != last_conbine):
            tot_medical += 1
        last_conbine = conbine
        M.add_word(re.sub(r'_', ' ', line.strip().lower()[:pos]), tot_medical)
        dict[5000+tot_medical] = line.strip().lower()[:pos]

C.make_automaton()
M.make_automaton()


# df_all = pd.read_csv('./papers.csv')
df_all = pd.read_csv('./papers_2022.csv')

start = 1
end = 6
start_pred = 259
end_pred = start_pred+5
df_all['Date'] = (df_all['year']-2000)*12+df_all['month']
print(tot_cs, "  ", tot_medical)
last_heat = np.zeros((tot_cs+100, tot_medical+100))
pred_heat = np.zeros((tot_cs+100, tot_medical+100))
train_data = {}
train_label = {}
test_data = {}
test_label = {}
for c in range(0, tot_cs+1):
    for m in range(0, tot_medical+1):
        train_data[c*10000+m] = []
        train_label[c*10000+m] = []
        test_data[c*10000+m] = []
        test_label[c*10000+m] = []
# while(end<=234):#2019-6
while(end <= end_pred):  # 2021-12
    print(start, "   ", end, len(train_label))
   # for c in range(1,tot_cs+1):
   #     for m in range(1,tot_medical+1):
   #         print(last_heat[c+20][m+20]," ")
    print("\n")
    df = df_all[(df_all['Date'] >= start) & (df_all['Date'] <= end)]
    df = df.reset_index(drop=True)
    # display(df)
    maxn = 0
    heat = np.zeros((tot_cs+100, tot_medical+100))
    for i in range(len(df)):
        # print(i)
        if(pd.isnull(df.loc[i]['abstract'])):
            continue
        str = df.loc[i]['abstract']
        medical_word = []
        cs_word = []
        for end_index, c in C.iter(df.loc[i]['abstract'].lower()):
            cs_word.append(c)
        for end_index, m in M.iter(df.loc[i]['abstract'].lower()):
            medical_word.append(m)
        for c in set(cs_word):
            for m in set(medical_word):
                heat[c+20, m+20] += 1
                if(heat[c+20, m+20] > maxn):
                    maxn = heat[c+20, m+20]
    # print(heat)
    if(start != 1):

        for c in range(0, tot_cs+1):
            for m in range(0, tot_medical+1):
                tmp = [start-6]
                for c1 in range(c-9, c+10):
                    for m1 in range(m-9, m+10):
                        tmp.append(last_heat[c1+20][m1+20])

                # if(start>=229): # 2019-01
                if(start >= start_pred):  # 2021-07
                    # print(train_data[c*10000+m])
                    # print(train_label[c*10000+m])
                    # print(tmp)
                    # print("---")

                    reg = Ridge(random_state=0).fit(np.asarray(
                        train_data[c*10000+m]), np.asarray(train_label[c*10000+m]))
                    pred_heat[c+20][m+20] = reg.predict(np.asarray([tmp]))[0]

                    if(pred_heat[c+20][m+20] < 0):
                        pred_heat[c+20][m+20] = 0

                else:
                    train_data[c*10000+m].append(tmp)
                    train_label[c*10000+m].append(heat[c+20, m+20])

    last_heat = heat
    start += 6
    end += 6
    
np.save("last_heat.npy", last_heat)
np.save("pred_heat.npy", pred_heat)

def get_r2(pred_heat, heat):
    y_true = []
    y_pred = []
    for c in range(0, tot_cs+1):
        for m in range(0, tot_medical+1):
            y_true.append(heat[c+20][m+20])
            y_pred.append(pred_heat[c+20][m+20])
    # print(y_true)
    # print(y_pred)

    # print("---------------------------------------")
    return r2_score(y_true, y_pred)


print("Ridge  r2_score:",  get_r2(pred_heat, heat))

# model = Word2Vec.load("word2vec/fulltext_abstract_phrases3.model")
model = Word2Vec.load("word2vec/fulltext_abstract_new.model")
row_dis = []
col_dis = []
for i in range(0, tot_cs+1):
    for j in range(i+1, tot_cs+1):
        row_dis.append(1-model.wv.similarity(dict[i], dict[j]))

for i in range(0, tot_medical+1):
    for j in range(i+1, tot_medical+1):
        col_dis.append(1-model.wv.similarity(dict[5000+i], dict[5000+j]))
row_linkage = hc.linkage(row_dis, method='average')
col_linkage = hc.linkage(col_dis, method='average')
row_order = hc.dendrogram(row_linkage)
col_order = hc.dendrogram(col_linkage)

with open('row_order.txt', 'w') as f:
    for i in range(len(row_order['leaves'])):
        f.write(dict[row_order['leaves'][i]]+'\n')

with open('col_order.txt', 'w') as f:
    for i in range(len(col_order['leaves'])):
        f.write(dict[col_order['leaves'][i]+5000]+'\n')



plot_heat = np.zeros((tot_cs+1, tot_medical+1))
for i in range(0, tot_cs+1):
    for j in range(0, tot_medical+1):
        plot_heat[i][j] = pred_heat[i+20][j+20]
plot_heat = np.log1p(plot_heat)

g=sns.clustermap(plot_heat, figsize=(20, 20), row_linkage=row_linkage, col_linkage=col_linkage, vmin=0, vmax=5,\
    yticklabels = [dict[r] for r in row_order['leaves']], xticklabels = [dict[5000 + c] for c in col_order['leaves']]) 

ax = g.ax_heatmap
ax.set_ylabel("cs_keywords")
ax.set_xlabel("medical_keywords")

ax.set_yticklabels(ax.get_ymajorticklabels(), fontdict={'fontsize':7})
ax.set_xticklabels(ax.get_xmajorticklabels(), fontdict={'fontsize':7})

# g.savefig("2019-6-pred_3.pdf")
g.savefig("picture/2021-12-pred_3.pdf")


for i in range(0, tot_cs+1):
    for j in range(0, tot_medical+1):
        plot_heat[i][j] = last_heat[i+20][j+20]
plot_heat = np.log1p(plot_heat)
g=sns.clustermap(plot_heat, figsize=(20, 20), row_linkage=row_linkage, col_linkage=col_linkage, vmin=0, vmax=5,\
    yticklabels = [dict[r] for r in row_order['leaves']], xticklabels = [dict[5000 + c] for c in col_order['leaves']]) 

ax = g.ax_heatmap
ax.set_ylabel("cs_keywords")
ax.set_xlabel("medical_keywords")

ax.set_yticklabels(ax.get_ymajorticklabels(), fontdict={'fontsize':7})
ax.set_xticklabels(ax.get_xmajorticklabels(), fontdict={'fontsize':7})
# g.savefig("2019-6-actal_3.pdf")
g.savefig("picture/2021-12-actal_3.pdf")
