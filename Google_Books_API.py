# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:54:42 2023

@author: gadge
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:40:48 2023

@author: gadge
"""

import os
import pandas as pd
from ast import literal_eval
import requests
import time


# Reading the new york times dataset
path = 'C:/Users/gadge/OneDrive/Desktop/MachineLearning/Project/Data'
df = pd.read_csv(path+'/NewYorkTimesBestsellersLists.csv')

df.head()

# Getting the title, author and isbn numbers list 
books_isbn13 = df[['title','author','isbn13']]

# The isbn13 column is a list of isbn numbers, expaning this to make a dataframe with isbn 
# number as the primary key
books_isbn13['isbn13'] = books_isbn13['isbn13'].apply(literal_eval)
books_isbn13 = books_isbn13.explode('isbn13')
books_isbn13 = books_isbn13.drop_duplicates()

books_isbn13.to_csv(path+'/Book_ISBN.csv',index = False)

# Getting these isbn numbers to query on google books, creating a dataframe and updating
# if it's available on google books api
isbn_list = books_isbn13['isbn13'][~(books_isbn13['isbn13'].isnull())]
isbn_list = pd.DataFrame(isbn_list).reset_index().drop('index',axis=1)
isbn_list['Google_API'] = 1

isbn_list.to_csv(path+'/isbn_list.csv',index = False)
isbn_list = pd.read_csv(path+'/isbn_list.csv')

isbn_remaining = isbn_list['isbn13'].tolist()

# Each Google API key has a very small limit of queries per day so using multiple API keys
APIs = ['AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXnU',
        'AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX20',
        'AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX28',
        'AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXSM',
        'AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxg',
        'AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXDo',
        'AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX2w',
        'AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXQE',
        'AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXL0',
        'AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX6k',
        'AIzaXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXOU']

api_index = 1
endpoint = 'https://www.googleapis.com/books/v1/volumes?'
Google_API = APIs[api_index]

#List of dictionaries with the details for isbn
isbns = []
i = 0
n = 0
for isbn in isbn_remaining:
    url = endpoint+'q=isbn:'+str(isbn)+'&key='+Google_API
    response = requests.get(url)
    jsontxt = response.json()
    # if the limit for an API is exceeded, the next one is picked from the list and
    # the ones with information collected is written to a csv file as a backup
    if 'error' in jsontxt.keys():
        isbn_df = pd.DataFrame(isbns) 
        isbn_df.to_csv(path+'/ISBN/isbn_data'+str(api_index)+'.csv',index = False)
        print("API Index",api_index)
        Google_API = APIs[api_index]
        api_index = api_index + 1
        url = endpoint+'q=isbn:'+str(isbn)+'&key='+Google_API
        response = requests.get(url)
        jsontxt = response.json()
    Items = jsontxt['totalItems']
    # if the book is not available on google books, the list is updated
    if 'items' not in jsontxt.keys():
        Items = 0
    if Items == 0:
        isbn_list.loc[isbn_list['isbn13']==isbn,'Google_API'] = 0
    # Creating a dictionary
    if Items > 0:
        item = jsontxt['items'][0]
        if 'subtitle' in item['volumeInfo'].keys():
            SubTitle = item['volumeInfo']['subtitle']
        else:
            SubTitle = 'N/A'
        if 'publisher' in item['volumeInfo'].keys():
            Publisher = item['volumeInfo']['publisher']
        else:
            Publisher = 'N/A'
        if 'publishedDate' in item['volumeInfo'].keys():
            Book_Published_date = item['volumeInfo']['publishedDate']
        else:
            Book_Published_date = 'N/A'
        if 'description' in item['volumeInfo'].keys():
            Description = item['volumeInfo']['description']
        else:
            Description = 'N/A'
        if 'pageCount' in item['volumeInfo'].keys():
            PageCount = item['volumeInfo']['pageCount']
        else:
            PageCount = 'N/A'
        Categories = item['volumeInfo']['categories'] if 'categories' in item['volumeInfo'] else 'N/A'
        Maturity = item['volumeInfo']['maturityRating'] if 'categories' in item['volumeInfo'] else 'N/A'
        Language = item['volumeInfo']['language'] if 'categories' in item['volumeInfo'] else 'N/A'
        Ebook_available = item['saleInfo']['isEbook'] if 'isEbook' in item['saleInfo'].keys() else 'N/A'
        if item['saleInfo']['saleability'] == 'NOT_FOR_SALE':
            list_price = 'Not For Sale'
            list_price_currency = 'Not For Sale'
            retail_price = 'Not For Sale'
            retail_price_currency = 'Not For Sale'
        else:
            list_price = item['saleInfo']['listPrice']['amount']
            list_price_currency = item['saleInfo']['listPrice']['currencyCode']
            retail_price = item['saleInfo']['retailPrice']['amount']
            retail_price_currency = item['saleInfo']['retailPrice']['currencyCode']
        Epub_available = item['accessInfo']['epub']['isAvailable'] if 'isAvailable' in item['accessInfo']['epub'].keys() else 'N/A'
        Epub_link_available = 'acsTokenLink' in item['accessInfo']['epub'].keys()
        pdf_available = item['accessInfo']['pdf']['isAvailable'] if 'isAvailable' in item['accessInfo']['pdf'].keys() else 'N/A'
        pdf_link_available = 'acsTokenLink' in item['accessInfo']['pdf'].keys()
        if 'searchInfo' in item.keys():
            TextSnippet = item['searchInfo']['textSnippet'] if 'textSnippet' in item['searchInfo'].keys() else 'N/A'
        else:
            TextSnippet = 'N/A'
        isbn_dict = {'isbn13':isbn,
                 'Items':Items,
                 'SubTitle':SubTitle,
                 'Publisher':Publisher,
                 'Book_Published_date':Book_Published_date,
                 'Description':Description,
                 'PageCount':PageCount,
                 'Categories':Categories,
                 'Maturity':Maturity,
                 'Language':Language,
                 'Ebook_available':Ebook_available,
                 'list_price':list_price,
                 'list_price_currency':list_price_currency,
                 'retail_price':retail_price,
                 'retail_price_currency':retail_price_currency,
                 'Epub_available':Epub_available,
                 'Epub_link_available':Epub_link_available,
                 'pdf_available':pdf_available,
                 'pdf_link_available':pdf_link_available,
                 'TextSnippet':TextSnippet}
        isbns.append(isbn_dict)
    i = i+1
    print(n*50+i)
    if i == 50:
        time.sleep(60)
        i = 0
        n = n+1
    

# Storing everything
isbn_df = pd.DataFrame(isbns) 

isbn_list.to_csv(path+'/isbn_list.csv',index = False)

isbn_df.to_csv(path+'/isbn_data.csv',index = False)
    
    
 

    
