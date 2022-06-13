search.txt                                  The searching command we using on Pubmed
pids.txt                                    All the pids for fulltext
fulltext-crawler.py                         get fulltext papers we need from pubmed
pubmed-crawler.py                           get abstract,keywords…… from pubmed
papers.csv                                  Data we get from above code
sorted_words.txt                            All the key words appear in abstract, we combine them into one-gram (like: Machine Learning -> Machine_Learning)
word2vec_batch.py                           train word2vec model from fulltext, abstract and PMC OA(more than 2 millions papers), when we are training, we combining all the words above
fulltext_abstract_phrases3.model            word2vec model we got
top200cs_combine.txt                        Top 100 frequency cs key word, with manually cleaning and combining 
top200medical_combine.txt                   Top 200 frequency medical key word, with manually cleaning and combining 
medical1000_combine.txt                     Top 500 frequency medical key word, with manually cleaning and combining 
cs1000_combine.txt                          Top 1000 frequency cs key word, with manually cleaning and combining 
heatmap_prediction_plot.py                  Get each pair of real value and predicted value
heatmap_clustermap.py                       Draw the cluster heatmap
heatmap_clustermap_2019_pred.py             Using all the previous data to predict the number of publications of each pair of CS and medical keywords, and plot the cluster cluster heatmap
2019-6-actal_3.pdf                          actual heatmap of 2019.1-2019.6
2019-6-pred_3.pdf                           predicted heatmap of 2019.1-2019.6
heatmap_prediction_all_by_real_sample_from_2010.py      Predict every years publications from 2010 to 2019 using all the previous data
heatmap_prediction_from_2015.py             Show how the R square decrease from 2015 if we using heatmap of 2015 and perdict iteratively
heatmap_top20_each_5years.py                To get TOP20 most popular CS and medical keywords every 5 years
heatmap_top20_in+decrease_each_5years.py    To get TOP20 increase and decrease of combination of CS&medical words
heatmap_top20_in+decrease_each_5years2.py   To get TOP20 increase and decrease of cs words
heatmap_top20_in+decrease_each_5years3.py   To get TOP20 increase and decrease of medical words
heatmap_top20_in+decrease_each_5years4.py   To get the number of publications every 5 years
in_decrease_for_cs.txt                      overall top 20 increasing and decreasing only for cs words every 5 years
in_decrease_for_med.txt                     overall top 20 increasing and decreasing only for medical words every 5 years
Number_of_publications_each_5_year.txt      the number of publications we get by searching keywords of "machine learning" "artificial intelligence" “classifier” “deep learning” "data mining" every 5 years
pubmed_figure.ipynb                         Jupyter notebook for generating all the figures




Using sklearn 0.19.1
The word2vec model are on the cluster since it was tooooo large
/dartfs-hpc/rc/lab/V/VosoughiS/lili/pub/fulltext_abstract_phrases3.model
/dartfs-hpc/rc/lab/V/VosoughiS/lili/pub/fulltext_abstract_phrases3.model.trainables.vectors_lockf.npy
/dartfs-hpc/rc/lab/V/VosoughiS/lili/pub/fulltext_abstract_phrases3.model.trainables.syn1neg.npy
/dartfs-hpc/rc/lab/V/VosoughiS/lili/pub/fulltext_abstract_phrases3.model.wv.vectors.npy 


# 265     270 Ridge  r2_score: -0.7222582107392554
# 259     264 Ridge  r2_score: 0.7745969672572482
different models, ridge, lasso
heatmap_prediction_all_by_real_sample_from_2010.py, not same as README description 2000 start

heatmap_clustermap.py, heatmap_clustermap_2019_pred.py, duplicate similar
heatmap_prediction_all_by_real_sample_from_2010.py  heatmap_prediction_from_2015.py similar

pubmed_figure why hard coded array, some window size have no spear

68.181.17.78, 68.181.17.*
571161
1553471
1560595