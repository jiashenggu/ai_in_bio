import gensim
from gensim.models import Word2Vec
import pandas as pd
import nltk
import re
import glob
from gensim.models.phrases import Phrases, Phraser
import ahocorasick
A = ahocorasick.Automaton()
nltk.download('punkt')

def not_all_uppercase(str):
    for i in range(len(str)):
        if(str[i]>='a' and str[i]<='z'):
            return True
    return False


with open("sorted_words.txt") as f:
    for line in f:
        if(line.find(' ')!=-1):
            A.add_word(line.strip().lower(),re.sub(r' ','_',line.strip().lower()))


A.make_automaton()




def not_all_uppercase(str):
    for i in range(len(str)):
        if(str[i]>='a' and str[i]<='z'):
            return True
    return False

tot=0
words=[]
df = pd.read_csv('./papers_2022.csv')
for i in range(len(df)):
    if(pd.isnull(df.loc[i]['abstract'])):
        continue
    str=df.loc[i]['abstract']
    for end_index, new_value in A.iter(df.loc[i]['abstract'].lower()):
        start_index = end_index - len(new_value) + 1
        str2=str[:start_index]+new_value+str[end_index+1:]
        str=str2
    sens= nltk.sent_tokenize(str)
    print(i)
    for sent in sens:
        new_sent=re.sub(r'[^a-zA-Z\-\_ ]','',sent)
        words_list=nltk.word_tokenize(new_sent)
        for i in range(len(words_list)):
            if not_all_uppercase(words_list[i]):
                words_list[i]=words_list[i].lower()
        #print(words_list)
        words.append(words_list)





paths = glob.glob("fulltext/*")
tot=0
for file_name in paths:
    tot=tot+1
    print("Full text No.",tot)
    file = open(file_name,"r",errors="ignore") 
    for line in file.readlines():
        newline=line.strip()
        if(newline!=""):
            if(newline[-1]=='.' or newline[-1]=='?' or newline[-1]=='!'):
                str=newline
                for end_index, new_value in A.iter(newline.lower()):
                    start_index = end_index - len(new_value) + 1
                    str2=str[:start_index]+new_value+str[end_index+1:]
                    str=str2
                sens= nltk.sent_tokenize(str)
                for sent in sens:
                    newline=re.sub(r'[^a-zA-Z\-\_ ]','',sent)
                    words_list=nltk.word_tokenize(newline)
                    for i in range(len(words_list)):
                        if not_all_uppercase(words_list[i]):
                            words_list[i]=words_list[i].lower()
                    #print(words_list)
                    words.append(words_list)
    file.close()

    
    
# paths = glob.glob("fulltext2/*")
# tot=0
# for file_name in paths:
#     tot=tot+1
#     print("Full text2 No.",tot)
#     file = open(file_name,"r",errors="ignore") 
#     for line in file.readlines():
#         newline=line.strip()
#         if(newline!=""):
#             if(newline[-1]=='.' or newline[-1]=='?' or newline[-1]=='!'):
#                 str=newline
#                 for end_index, new_value in A.iter(newline.lower()):
#                     start_index = end_index - len(new_value) + 1
#                     str2=str[:start_index]+new_value+str[end_index+1:]
#                     str=str2
#                 sens= nltk.sent_tokenize(str)
#                 for sent in sens:
#                     newline=re.sub(r'[^a-zA-Z\-\_ ]','',sent)
#                     words_list=nltk.word_tokenize(newline)
#                     for i in range(len(words_list)):
#                         if not_all_uppercase(words_list[i]):
#                             words_list[i]=words_list[i].lower()
#                     #print(words_list)
#                     words.append(words_list)
#     file.close()
# model = Word2Vec(size=300,  min_count=1, window=5)
model = Word2Vec.load('/home/jiasheng/ai_in_bio/word2vec/fulltext_abstract_phrases3.model')
model.build_vocab(words, update=True)
model.train(words, total_examples=len(words), epochs=5)
model.save("word2vec/fulltext_abstract_new.model")
# words=[]


# paths = glob.glob("fulltext_oa/*/*.txt")
# tot=0
# for file_name in paths:
#     tot=tot+1
#     print(file_name)
#     file = open(file_name,"r",errors="ignore") 
#     for line in file.readlines():
#         newline=line.strip()
#         if(newline!=""):
#             if(newline[-1]=='.' or newline[-1]=='?' or newline[-1]=='!'):
#                 str=newline
#                 for end_index, new_value in A.iter(newline.lower()):
#                     start_index = end_index - len(new_value) + 1
#                     str2=str[:start_index]+new_value+str[end_index+1:]
#                     str=str2
#                 sens= nltk.sent_tokenize(str)
#                 for sent in sens:
#                     newline=re.sub(r'[^a-zA-Z\-\_ ]','',sent)
#                     words_list=nltk.word_tokenize(newline)
#                     for i in range(len(words_list)):
#                         if not_all_uppercase(words_list[i]):
#                             words_list[i]=words_list[i].lower()
#                     #print(words_list)
#                     words.append(words_list)
#     if(tot>100000):
#         print(tot)
#         tot=0
#         model.build_vocab(words,update=True)
#         model.train(words,total_examples=len(words), epochs=model.iter)
#         words=[]
#     file.close()
# print("hahahhhahahahahah")
    
# paths = glob.glob("fulltext_oa_non/*/*.txt")
# for file_name in paths:
#     tot=tot+1
#     print(file_name)
#     file = open(file_name,"r",errors="ignore") 
#     for line in file.readlines():
#         newline=line.strip()
#         if(newline!=""):
#             if(newline[-1]=='.' or newline[-1]=='?' or newline[-1]=='!'):
#                 str=newline
#                 for end_index, new_value in A.iter(newline.lower()):
#                     start_index = end_index - len(new_value) + 1
#                     str2=str[:start_index]+new_value+str[end_index+1:]
#                     str=str2
#                 sens= nltk.sent_tokenize(str)
#                 for sent in sens:
#                     newline=re.sub(r'[^a-zA-Z\-\_ ]','',sent)
#                     words_list=nltk.word_tokenize(newline)
#                     for i in range(len(words_list)):
#                         if not_all_uppercase(words_list[i]):
#                             words_list[i]=words_list[i].lower()
#                     #print(words_list)
#                     words.append(words_list)
#     if(tot>100000):
#         print(tot)
#         tot=0
#         model.build_vocab(words,update=True)
#         model.train(words,total_examples=len(words), epochs=model.iter)
#         words=[]
#     file.close()




# model.build_vocab(words,update=True)
# model.train(words,total_examples=len(words), epochs=model.iter)
# model.save("fulltext_abstract_phrases3.model")
