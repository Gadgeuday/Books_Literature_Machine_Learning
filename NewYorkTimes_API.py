# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import requests
import re
import time

Path = 'C:/Users/gadge/OneDrive/Desktop/MachineLearning/Project'

nyt_api = "XXXXXXXXXXXXXXXXXXXXX"


lists = ['combined-print-and-e-book-fiction',
 'combined-print-and-e-book-nonfiction',
 'hardcover-fiction',
 'hardcover-nonfiction',
 'trade-fiction-paperback',
 'paperback-nonfiction',
 'advice-how-to-and-miscellaneous',
 'young-adult-hardcover']

endpoint = 'https://api.nytimes.com/svc/books/v3/lists/'


date = '2023-02-12'

while int(date[0:4]) > 2009:
    time.sleep(60)
    books = []
    for genre in lists:
        url = endpoint+date+'/'+genre+'.json?api-key='+nyt_api
        response = requests.get(url)
        jsontxt = response.json()
        published_date = jsontxt['results']['published_date']
        previous_date = jsontxt['results']['previous_published_date']
        for book in jsontxt['results']['books']:
            book_dict = {'genre':genre,'published_date':published_date}
            book_dict['rank'] = book['rank']
            book_dict['rank_last_week'] = book['rank_last_week']
            book_dict['weeks_on_list'] = book['weeks_on_list']
            book_dict['publisher'] = book['publisher']
            book_dict['description'] = book['description']
            book_dict['price'] = book['price']
            book_dict['title'] = book['title']
            book_dict['author'] = book['author']
            books.append(book_dict)

    print(previous_date)
    print(published_date)
        
    date = previous_date
        
    book_list = [i['title'] for i in books]
        #print(book_list)
        
    books_df = pd.DataFrame(books)
        
    file_name = 'books_data_'+published_date+'.csv'
        
    books_df.to_csv(Path+'/Data/NYTimes_Bestsellers/'+file_name,index=False)

### date = '2022-10-02'


