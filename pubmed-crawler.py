from cmath import pi
import sys
import pubmed_parser as pp
import csv
import time
import re
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
}

with open("pids_new.txt") as f:
    str_pids = f.readlines()

pids = [int(x.strip()) for x in str_pids]


# pids = []
# flag = False
# for x in str_pids:
#     pid = int(x.strip())
#     if pid == 35079207:
#         flag = True
#     if flag:
#         pids.append(pid)
print(pids)
        

papers = open("papers_2022.csv", "w", encoding='utf-8')
papers.write("pmid,title,abstract,journal,affiliation,authors,keywords,doi,year,month,citations,keywords_match\n")
for i in range(len(pids)):
    print(i)
    try:
        dict_out = pp.parse_xml_web(pids[i], save_xml=False)
        papers.write('"' + re.sub(r'"', "", dict_out['pmid']) + '",')
        papers.write('"' + re.sub(r'"', "", dict_out['title']) + '",')
        papers.write('"' + re.sub(r'"', "", dict_out['abstract']) + '",')
        papers.write('"' + re.sub(r'"', "", dict_out['journal']) + '",')
        papers.write('"' + re.sub(r'"', "", dict_out['affiliation']) + '",')
        papers.write('"' + re.sub(r'"', "", dict_out['authors']) + '",')
        papers.write('"' + re.sub(r'"', "", dict_out['keywords']) + '",')
        papers.write('"' + re.sub(r'"', "", dict_out['doi']) + '",')
        papers.write('"' + re.sub(r'"', "", dict_out['year']) + '",')

        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=" + dict_out[
            'pmid'] + "&tool=my_tool&email=my_email@example.com"
        response = requests.get(url, headers=headers)
        searchObj = re.search(r'month ([0-9]+)', response.text, re.M)

        if searchObj:
            papers.write('"' + searchObj.group(1) + '",')
        else:
            papers.write('"",')

        url = "https://scholar.google.com/scholar?hl=en&as_sdt=0%2C46&q=" + dict_out['doi'] + "&btnG="
        response = requests.get(url, headers=headers)
        searchObj = re.search(r'Cited by ([0-9]+)', response.text, re.M)
        if searchObj:
            papers.write('"' + searchObj.group(1) + '",')
        else:
            papers.write('"0",')

        keywords = ""
        searchObj = re.search(r'machine.learning', dict_out['title'] + dict_out['abstract'] + dict_out['keywords'],
                              re.M | re.I)
        if searchObj:
            keywords += "Machine Learning;"

        searchObj = re.search(r'deep.learning', dict_out['title'] + dict_out['abstract'] + dict_out['keywords'],
                              re.M | re.I)
        if searchObj:
            keywords += "Deep Learning;"

        searchObj = re.search(r'data.science', dict_out['title'] + dict_out['abstract'] + dict_out['keywords'],
                              re.M | re.I)
        if searchObj:
            keywords += "Data Science;"

        searchObj = re.search(r'artificial.intelligence',
                              dict_out['title'] + dict_out['abstract'] + dict_out['keywords'], re.M | re.I)
        if searchObj:
            keywords += "Artificial Intelligence;"

        searchObj = re.search(r'classifier', dict_out['title'] + dict_out['abstract'] + dict_out['keywords'],
                              re.M | re.I)
        if searchObj:
            keywords += "Classifier;"

        papers.write('"' + keywords + '"\n')
        time.sleep(1)

    except Exception as e:
        print(pids[i])
        print(e)
        papers.write('\n')
        continue
