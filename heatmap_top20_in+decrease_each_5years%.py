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
        dict[tot_cs] = conbine


with open("medical1000_combine.txt") as f:
    for line in f:
        pos = line.strip().find(':')
        conbine = re.sub(r'_', ' ', line.strip().lower()[pos+1:])
        if(conbine != last_conbine):
            tot_medical += 1
        last_conbine = conbine
        M.add_word(re.sub(r'_', ' ', line.strip().lower()[:pos]), tot_medical)
        dict[5000+tot_medical] = conbine


C.make_automaton()
M.make_automaton()

df_all = pd.read_csv('./papers.csv')
start = 1 + 8*60
end = 60 + 8*60
df_all['Date'] = (df_all['year']-1970)*12+df_all['month']

print(tot_cs, "  ", tot_medical)
last_heat = np.zeros((tot_cs+100, tot_medical+100))

# while(end<=600):#234: 2019-12
while(end <= 624):  # 2021-12
    print(start, "   ", end)
    # for c in range(1,tot_cs+1):
    #    for m in range(1,tot_medical+1):
    #        print(last_heat[c+20][m+20]," ")
    # print("\n")
    df = df_all[(df_all['Date'] >= start) & (df_all['Date'] <= end)]
    df = df.reset_index(drop=True)
    # display(df)
    maxn = 0
    heat = np.zeros((tot_cs+100, tot_medical+100))
    cs = np.zeros(tot_cs+100)
    medical = np.zeros(tot_medical+100)
    for i in range(len(df)):
        # print(i)
        if(pd.isnull(df.loc[i]['abstract'])):
            continue
        medical_word = []
        cs_word = []
        for end_index, c in C.iter(df.loc[i]['abstract'].lower()):
            cs_word.append(c)
        for end_index, m in M.iter(df.loc[i]['abstract'].lower()):
            medical_word.append(m)
        for c in set(cs_word):
            cs[c] += 1
        for m in set(medical_word):
            medical[m] += 1
        for c in set(cs_word):
            for m in set(medical_word):
                heat[c, m] += 1
    print("___________________________________________________________")
    print("Result of Year  ", int(start/12)+1970, "  :")

    heats = []
    names = []
    total = 0
    for c in range(0, tot_cs+1):
        for m in range(0, tot_medical+1):
            # inc = heat[c,m]-last_heat[c,m]
            if last_heat[c, m] == 0:
                continue
            inc = (heat[c,m]-last_heat[c,m])/last_heat[c, m]*100
            total += inc
            heats.append(inc)
            names.append(dict[c]+" , "+dict[m+5000])
    if np.sum(heats)==0:
        start += 60
        end += 60
        last_heat = heat
        continue
    tot_len = len(heats)
    print(len(heats), len(names))
    # for i in range(tot_len):
    #     heats[i] /= total
    #     heats[i] *= 100
    heats, names = zip(*sorted(zip(heats, names)))

    for t, n in zip(heats, names):
        print(t, n)

    print("TOP20 increase of cs&medical:")
    inc_names = []
    inc_heats = []
    for i in range(20):
        inc_names.append(names[len(names)-1-i])
        inc_heats.append(heats[len(names)-1-i])
    df = pd.DataFrame({'name': inc_names, '%': inc_heats})
    df.index = range(1, 21)
    print(df.to_latex(float_format="%.3f"))
    print(" ")

    print("TOP20 decrease of cs&medical:")
    dec_names = []
    dec_heats = []
    for i in range(20):
        dec_names.append(names[i])
        dec_heats.append(heats[i])
    df = pd.DataFrame({'name': dec_names, '%': dec_heats})
    df.index = range(1, 21)
    print(df.to_latex(float_format="%.3f"))
    print(" ")
    start += 60
    end += 60
    last_heat = heat