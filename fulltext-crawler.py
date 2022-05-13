import sys
import pubmed_parser as pp
import csv
import time
import re
import requests
from bs4 import BeautifulSoup
import pprint
from lxml import html


def convert_document_id(doc_id, id_type='PMID'):
    """
    Convert document id to dictionary of other id
    see: http://www.ncbi.nlm.nih.gov/pmc/tools/id-converter-api/ for more info
    """
    doc_id = str(doc_id)
    if id_type == 'PMC':
        doc_id = 'PMC%s' % doc_id
        pmc = doc_id
        convert_link = 'http://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids=%s' % doc_id
    elif id_type in ['PMID', 'DOI', 'OTHER']:
        convert_link = 'http://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids=%s' % doc_id
    else:
        raise ValueError('Give id_type from PMC or PMID or DOI or OTHER')

    convert_page = requests.get(convert_link)
    convert_tree = html.fromstring(convert_page.content)
    record = convert_tree.find('record').attrib
    if 'status' in record or 'pmcid' not in record:
        raise ValueError('Cannot convert given document id to PMC')
    if id_type in ['PMID', 'DOI', 'OTHER']:
        if 'pmcid' in record:
            pmc = record['pmcid']
        else:
            pmc = ''
    return pmc


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36',
}

with open("pids_new.txt") as f:
    str1 = f.readlines()

pids = [int(x.strip()) for x in str1]
tot = 0
for i in range(len(pids)):
    try:
        tot = tot + 1
        print(tot)
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=" + convert_document_id(int(pids[i]),
                                                                                                           id_type='PMID') + "&tool=my_tool&email=my_email@example.com"
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        period = soup.find('pmc-articleset').find('body').get_text()
        f = open("fulltext/" + str(pids[i]), "w", encoding='utf-8')
        f.write(period)
        f.write("\n")
        time.sleep(1)
        f.close()
    except Exception as e:
        print(e)
        continue
