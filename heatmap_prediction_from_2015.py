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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr
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
with open("cs1000_combine.txt") as f:
    for line in f:
        pos = line.strip().find(':')
        conbine = re.sub(r'_', ' ', line.strip().lower()[pos+1:])
        if(conbine != last_conbine):
            tot_cs += 1
        last_conbine = conbine
        C.add_word(re.sub(r'_', ' ', line.strip().lower()[:pos]), tot_cs)

        print(line.strip().lower()[:pos], "    ", conbine, "    ", tot_cs)

with open("medical1000_combine.txt") as f:
    for line in f:
        pos = line.strip().find(':')
        conbine = re.sub(r'_', ' ', line.strip().lower()[pos+1:])
        if(conbine != last_conbine):
            tot_medical += 1
        last_conbine = conbine
        M.add_word(re.sub(r'_', ' ', line.strip().lower()[:pos]), tot_medical)


def get_r2(pred_heat, heat):
    y_true = []
    y_pred = []
    for c in range(1, tot_cs+1):
        for m in range(1, tot_medical+1):
            y_true.append(heat[c+20][m+20])
            y_pred.append(pred_heat[c+20][m+20])
    #print(y_true)
    #print(y_pred)

    #print("---------------------------------------")
    return r2_score(y_true, y_pred)


def get_spearmanr(pred_heat, heat):
    y_true = []
    y_pred = []
    for c in range(1, tot_cs+1):
        for m in range(1, tot_medical+1):
            y_true.append(heat[c+20][m+20])
            y_pred.append(pred_heat[c+20][m+20])
    #print(y_true)
    #print(y_pred)

    #print("---------------------------------------")
    return spearmanr(y_true, y_pred).correlation


C.make_automaton()
M.make_automaton()

# df_all = pd.read_csv('./papers.csv')
df_all = pd.read_csv('./papers_2022.csv')
start = 1
end = 6
df_all['Date'] = (df_all['year']-2000)*12+df_all['month']
start = 1
end = 6
print(tot_cs, "  ", tot_medical)
last_heat = np.zeros((tot_cs+100, tot_medical+100))
pred_heat = np.zeros((tot_cs+100, tot_medical+100))
train_data = {}
train_label = {}
test_data = {}
test_label = {}
for c in range(1, tot_cs+1):
    for m in range(1, tot_medical+1):
        train_data[c*10000+m] = []
        train_label[c*10000+m] = []
        test_data[c*10000+m] = []
        test_label[c*10000+m] = []
results = {}
results['date'] = []

results['lasso_r2'] = []
results['lasso_spearmanr'] = []
# while(end<=234):#2019-6
while(end <= 264):  # 2021-12
    print(start, "   ", end, len(train_label))
   # for c in range(1,tot_cs+1):
   #     for m in range(1,tot_medical+1):
   #         print(last_heat[c+20][m+20]," ")
    print("\n")
    df = df_all[(df_all['Date'] >= start) & (df_all['Date'] <= end)]
    df = df.reset_index(drop=True)
    #display(df)
    maxn = 0
    heat = np.zeros((tot_cs+100, tot_medical+100))
    for i in range(len(df)):
        #print(i)
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
    #print(heat)
    if(start != 1):

        for c in range(1, tot_cs+1):
            for m in range(1, tot_medical+1):
                tmp = [start-6]
                for c1 in range(c-1, c+2):
                    for m1 in range(m-1, m+2):
                        tmp.append(last_heat[c1+20][m1+20])

                if(start >= 175):
                    #print(train_data[c*10000+m])
                    #print(train_label[c*10000+m])
                    #print(tmp)
                    #print("---")
                    reg = Lasso(random_state=0).fit(np.asarray(
                        train_data[c*10000+m]), np.asarray(train_label[c*10000+m]))
                    pred_heat[c+20][m+20] = reg.predict(np.asarray([tmp]))[0]
                    if(pred_heat[c+20][m+20] < 0):
                        pred_heat[c+20][m+20] = 0

                else:
                    train_data[c*10000+m].append(tmp)
                    train_label[c*10000+m].append(heat[c+20, m+20])
        lasso_r2 = get_r2(pred_heat, heat)
        lasso_spearmanr = get_spearmanr(pred_heat, heat)
        print(start/12, " ", start % 12, "  r2_score:",
              lasso_r2, "   ", lasso_spearmanr)

        results['date'].append("{}/{}".format(2000+int(start/12), start % 12))
        results['lasso_r2'].append(lasso_r2)
        results['lasso_spearmanr'].append(lasso_spearmanr)

    last_heat = heat
    if(start >= 175):
        last_heat = pred_heat
    start += 6
    end += 6


pd.DataFrame(results).to_csv('lasso_cycle_5_2022.csv')
