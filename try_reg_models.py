from curses import window
from locale import normalize
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
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.linear_model import ElasticNet, PassiveAggressiveRegressor, TheilSenRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import warnings

warnings.filterwarnings('ignore')

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
        
        print(line.strip().lower()[:pos],"    ",conbine,"    ",tot_cs)

with open("medical1000_combine.txt") as f:
    for line in f:
        pos=line.strip().find(':')
        conbine=re.sub(r'_',' ',line.strip().lower()[pos+1:])
        if(conbine!=last_conbine):
            tot_medical+=1
        last_conbine=conbine
        M.add_word(re.sub(r'_',' ',line.strip().lower()[:pos]),tot_medical)

def get_r2(pred_heat,heat):
    y_true=[]
    y_pred=[]
    for c in range(1,tot_cs+1):
        for m in range(1,tot_medical+1):
            y_true.append(heat[c+20][m+20])
            y_pred.append(pred_heat[c+20][m+20])
    #print(y_true)
    #print(y_pred)

    #print("---------------------------------------")
    return r2_score(y_true, y_pred)
    
def get_spearmanr(pred_heat,heat):
    y_true=[]
    y_pred=[]
    for c in range(1,tot_cs+1):
        for m in range(1,tot_medical+1):
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
start=1
end=6
df_all['Date']=(df_all['year']-2000)*12+df_all['month']
start=1
end=6
print(tot_cs,"  ",tot_medical)
last_heat=np.zeros((tot_cs+100,tot_medical+100))
pred_ridge_false=np.zeros((tot_cs+100,tot_medical+100))
pred_ridge_true=np.zeros((tot_cs+100,tot_medical+100))
pred_svr=np.zeros((tot_cs+100,tot_medical+100))
pred_lasso=np.zeros((tot_cs+100,tot_medical+100))

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

results = {}
results['date'] = []




try_model = "OMP"
heats = {}
pred_ridge_false = {}
pred_ridge_true = {}
pred_svr = {}
pred_lasso = {}
start = 1
end = 6
# while(end<=234):#2019-6
while(end<=264):#2021-12
    date = "{}/{}".format(2000+int(start/12), start%12)
    
    df=df_all[ (df_all['Date']>=start) &(df_all['Date']<=end)]
    df = df.reset_index(drop=True)
    #display(df)

    heat = np.zeros((tot_cs+100,tot_medical+100))

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
                heat[c+20, m+20]+=1
    heats[date] = heat
    if start!=1:
        print(date, start)
        results['date'].append(date)
        pred_ridge_false[date]=np.zeros((tot_cs+100,tot_medical+100))
        pred_ridge_true[date]=np.zeros((tot_cs+100,tot_medical+100))
        pred_svr[date]=np.zeros((tot_cs+100,tot_medical+100))
        pred_lasso[date]=np.zeros((tot_cs+100,tot_medical+100))
    start += 6
    end += 6

print("---start---")
# window_size = 9
for window_size in [1, 3, 5, 7, 9]:
    results[try_model] = []
    print("window_size: {}".format(window_size))
    for cc in range(1,tot_cs+1):
        print(cc)
        for mm in range(1,tot_medical+1):
            train_data=[]
            train_label=[]
            start = 1
            end = 6
            # while(end<=234):#2019-6
            while(end<=264):#2021-12
                # print(start,"   ", end, len(train_label))
                date = "{}/{}".format(2000+int(start/12), start%12)
            # for c in range(1,tot_cs+1):
            #     for m in range(1,tot_medical+1):
            #         print(last_heat," ")
                heat = heats[date]

                #print(heat)
                if(start!=1):
                    layers=window_size
                    # for c in range(1,tot_cs+1):
                    #     for m in range(1,tot_medical+1):
                    tmp=[start-6] # why
                    for c1 in range(cc-layers+1,cc+layers):
                        for m1 in range(m-layers+1,m+layers):
                            tmp.append(last_heat[c1+20, m1+20])
                            
                    if(start>=30):
                        #print(train_data)
                        #print(train_label)
                        #print(tmp)
                        #print("---")

                            
                            
                        reg = OrthogonalMatchingPursuit().fit(np.asarray(train_data), np.asarray(train_label))
                        pred_svr[date][cc+20, mm+20]=reg.predict(np.asarray([tmp]))[0]
                        if(pred_svr[date][cc+20, mm+20]<0):
                            pred_svr[date][cc+20, mm+20]=0
                            

                            
                    train_data.append(tmp)
                    train_label.append(heat[cc+20, mm+20])

                    # print(2000+int(start/12)," ",start%12,"  :\n")
                    # print(get_r2(pred_lr,heat)," ", get_spearmanr(pred_lr,heat))
                    # print(get_r2(pred_svr,heat)," ", get_spearmanr(pred_svr,heat))
                    # print(get_r2(pred_lasso,heat)," ", get_spearmanr(pred_lasso,heat))
                    # print(get_r2(pred_ridge_false,heat)," ", get_spearmanr(pred_ridge_false,heat))
                    # print(get_r2(pred_ridge_true,heat)," ", get_spearmanr(pred_ridge_true,heat))
                    # results['date'].append("{}/{}".format(2000+int(start/12), start%12))
                    # results['linear_regression_r2'].append(get_r2(pred_lr, heat))

                    # results['linear_regression_spearmanr'].append(get_spearmanr(pred_lr, heat))
                    # results['svr_spearmanr'].append(get_spearmanr(pred_svr, heat))
                    # results['lasso_spearmanr'].append(get_spearmanr(pred_lasso, heat))
                    # results['ridge_false_spearmanr'].append(get_spearmanr(pred_ridge_false, heat))
                    # results['ridge_true_spearmanr'].append(get_spearmanr(pred_ridge_true, heat))
                last_heat=heat
                start+=6
                end+=6
    for date in results['date']:
        results[try_model].append(get_r2(pred_svr[date], heats[date]))
    pd.DataFrame(results).to_csv('window_size_{}_OrthogonalMatchingPursuit.csv'.format(window_size))








