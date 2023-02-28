# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:07:34 2023

@author: gadge
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import requests

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

Path = 'C:/Users/gadge/OneDrive/Desktop/MachineLearning/Project'

file_path = Path + '/Data'
df= pd.read_csv(file_path+"/NewYorkTimesBestsellersLists.csv")

df['author'] = df['author'].apply(lambda x: re.sub(r'edited ',"",x))
df['authors'] = df['author'].apply(lambda x: re.split(r'\b and \b|\b with \b|\,', x))
df['#authors'] = df['authors'].apply(len)

df['primary_author'] = df['authors'].apply(lambda x: x[0])
df['secondary_author'] = df['authors'].apply(lambda x: x[1] if len(x)>1 else None)


books = df['title'].unique().tolist()
books

def cased_text(text):
    return "_".join([i[0]+i[1:].lower() for i in text.split(" ") if len(i) != 0])

wikipedia_base = 'https://en.wikipedia.org/wiki/'

book_info = []
for book in books:
    print(book)
    genre = 'N/A'
    author = 'N/A'
    original_publication = 'N/A'
    url = wikipedia_base+cased_text(book)
    website = requests.get(url)
    if website.status_code == 200:
        soup = BeautifulSoup(website.text, 'html.parser')
        for link in soup.find_all('table'):
            if link.get('class'):
                if len(link.get('class')) > 1:
                    if ((link.get('class')[0] == 'infobox')&(link.get('class')[1] == 'vcard')):
                        for value in link.find_all('tr'):
                            if value.find('th') is not None:
                                if value.find('th').text == 'Author':
                                    author = value.find('td').text
                                if value.find('th').text == 'Genre':
                                    try:
                                        genre = [i.text for i in value.find_all('a')]
                                    except:
                                        genre = value.find('td').text
                                if len(genre) == 0:
                                    genre = value.find('td').text
                                if value.find('th').text == 'Publication date':
                                    original_publication = value.find('td').text
    book_dict = {'book':book,'Wiki_author':author,'Genre':genre,"original_publication":original_publication}
    book_info.append(book_dict)
    
books_extracted = [i['book'] for i in book_info if i['Wiki_author'] != 'N/A']
books_remaining = [i for i in books if i not in books_extracted]

import wikipedia

book_info_1 = []
for book in books_remaining:
    #print(book)
    genre = 'N/A'
    author = 'N/A'
    original_publication = 'N/A'
    try:
        results = wikipedia.search(book)
        url = wikipedia.page(results[0]).url
        website = requests.get(url)
        print(book)
        if website.status_code == 200:
            soup = BeautifulSoup(website.text, 'html.parser')
            for link in soup.find_all('table'):
                if link.get('class'):
                    if len(link.get('class')) > 1:
                        if ((link.get('class')[0] == 'infobox')&(link.get('class')[1] == 'vcard')):
                            for value in link.find_all('tr'):
                                if value.find('th') is not None:
                                    if value.find('th').text == 'Author':
                                        author = value.find('td').text
                                    if value.find('th').text == 'Genre':
                                        try:
                                            genre = [i.text for i in value.find_all('a')]
                                        except:
                                            genre = value.find('td').text
                                    if len(genre) == 0:
                                        genre = value.find('td').text
                                    if value.find('th').text == 'Publication date':
                                        original_publication = value.find('td').text
        book_dict = {'book':book,'Wiki_author':author,'Genre':genre,"original_publication":original_publication}
        book_info_1.append(book_dict)
    except:
        book_dict = {'book':book,'Wiki_author':author,'Genre':genre,"original_publication":original_publication}
        book_info_1.append(book_dict)

books_extracted_dict = [i for i in book_info if i['Wiki_author'] != 'N/A']
total_books = books_extracted_dict + book_info_1

total_books_wiki_df = pd.DataFrame(total_books)
total_books_wiki_df.to_excel(file_path+'/Wikipedia_information.xlsx',index = False)


