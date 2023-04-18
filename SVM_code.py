# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:51:44 2023

@author: gadge
"""


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
%matplotlib inline

path = 'C:/Users/gadge/OneDrive/Desktop/MachineLearning/Project/Data'
NYT = pd.read_csv(path+'/NewYorkTimesBestsellersLists.csv')

Fiction = ["combined-print-and-e-book-fiction","hardcover-fiction",
             "trade-fiction-paperback"]
Non_Fiction = ["combined-print-and-e-book-nonfiction","hardcover-nonfiction",
                 "paperback-nonfiction"]
Advice = ["advice-how-to-and-miscellaneous"]
Young_adult = ["young-adult-hardcover"]

def genre_def(text):
    if text in Fiction:
        return "Fiction"
    elif text in Non_Fiction:
        return "Non Fiction"
    elif text in Advice:
        return "Advice"
    else:
        return "Young Adult"
    
NYT['Genre'] = NYT['genre'].apply(genre_def)

NYT = NYT[['Genre','description','title','author']]

NYT = NYT[~(NYT['description'].isnull())]

NYT = NYT.drop_duplicates(subset = ['title','author'], keep = 'first')

NYT['Genre'].value_counts(normalize = True)

NYT = NYT[['Genre','description']]

lemmatizer = WordNetLemmatizer()
def text_clean(data):
    data = re.sub( '[^A-Za-z]', ' ', data)
    data = re.sub(' +', ' ', data).lower()
    word_list = word_tokenize(data)
    data = ' '.join([lemmatizer.lemmatize(w) for w in word_list if len(w) > 3])
    return data

NYT['description'] = NYT['description'].astype('str').apply(text_clean)

NYT.isnull().sum()

NYT = NYT[NYT['description'] != 'nan']

### Wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import random
path = 'C:/Users/gadge/OneDrive/Desktop/MachineLearning/Project/plots'

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % 30

## Wordclouds
for genre in NYT['Genre'].unique():
    print(genre)
    text = NYT[NYT['Genre'] == genre]['description'].apply(lambda x: ' '.join([i for i in x.split(" ") if (len(i) > 4)]))
    text = ' '.join([i for i in text.tolist() if len(i) > 3])

    wordcloud = WordCloud(stopwords = STOPWORDS,
                          collocations=True,
                          width = 1000, height = 1000,
                          background_color ='white',
                          max_words = 120).generate(text)



    plt.imshow(wordcloud.recolor(color_func=grey_color_func, random_state=3),
               interpolation='bilInear')
    plt.axis('off')
    plt.savefig(path+'/wordcloud_'+genre+'.png',format = 'png')
    plt.show()

vectorizer = TfidfVectorizer(stop_words = 'english',max_df=0.8)
X = vectorizer.fit_transform(NYT['description'])

words = vectorizer.get_feature_names()

X = X.toarray()

y = NYT['Genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
path = 'C:/Users/gadge/OneDrive/Desktop/MachineLearning/Project/plots'
### Linear SVC C = 10
from sklearn.svm import LinearSVC
SVM_Linear1 =LinearSVC(C=10)
SVM_Linear1.fit(X_train, y_train)
y_pred = SVM_Linear1.predict(X_test)

cnf_matrix1 = confusion_matrix(y_test, y_pred, labels = SVM_Linear1.classes_)
print("\nThe confusion matrix is:")
print(cnf_matrix1)
ax = sns.heatmap(cnf_matrix1, annot=True, fmt='d' )
ax.xaxis.set_ticklabels(SVM_Linear1.classes_)
ax.yaxis.set_ticklabels(SVM_Linear1.classes_)
plt.title("Linear SVC with C = 10")
plt.savefig(path+'/Linear_SVC_C_10.png',format = 'png')

print(classification_report(y_test,y_pred))

### Linear SVC C = 50
from sklearn.svm import LinearSVC
SVM_linear2=LinearSVC(C=50)
SVM_linear2.fit(X_train, y_train)
y_pred = SVM_linear2.predict(X_test)

cnf_matrix1 = confusion_matrix(y_test, y_pred, labels = SVM_linear2.classes_)
print("\nThe confusion matrix is:")
print(cnf_matrix1)
ax = sns.heatmap(cnf_matrix1, annot=True, fmt='d' )
ax.xaxis.set_ticklabels(SVM_linear2.classes_)
ax.yaxis.set_ticklabels(SVM_linear2.classes_)
plt.title("Linear SVC with C = 50")
plt.savefig(path+'/Linear_SVC_C_50.png',format = 'png')

print(classification_report(y_test,y_pred))
## RBF and cost 1
from sklearn.svm import SVC
SVM_rbf = SVC(C = 1,kernel='rbf')
SVM_rbf.fit(X_train, y_train)

y_pred = SVM_rbf.predict(X_test)

cnf_matrix1 = confusion_matrix(y_test, y_pred, labels = SVM_rbf.classes_)
print("\nThe confusion matrix is:")
print(cnf_matrix1)
ax = sns.heatmap(cnf_matrix1, annot=True, fmt='d' )
ax.xaxis.set_ticklabels(SVM_rbf.classes_)
ax.yaxis.set_ticklabels(SVM_rbf.classes_)
plt.title("SVM with rbf kernel with C = 1")
plt.savefig(path+'/rbf_SVC_C_1.png',format = 'png')

print(classification_report(y_test,y_pred))
### 

## poly and cost 10
from sklearn.svm import SVC
SVM_poly1 = SVC(C = 10,kernel='poly',degree = 4)
SVM_poly1.fit(X_train, y_train)

y_pred = SVM_poly1.predict(X_test)

cnf_matrix1 = confusion_matrix(y_test, y_pred, labels = SVM_poly1.classes_)
print("\nThe confusion matrix is:")
print(cnf_matrix1)
ax = sns.heatmap(cnf_matrix1, annot=True, fmt='d' )
ax.xaxis.set_ticklabels(SVM_poly1.classes_)
ax.yaxis.set_ticklabels(SVM_poly1.classes_)
plt.title("SVM with Polynomial kernel with C = 10 and degree 4")
plt.savefig(path+'/rbf_poly_C_10.png',format = 'png')

print(classification_report(y_test,y_pred))
### 

## poly and cost 1
from sklearn.svm import SVC
SVM_poly1 = SVC(C = 1,kernel='poly',degree = 3)
SVM_poly1.fit(X_train, y_train)

y_pred = SVM_poly1.predict(X_test)

cnf_matrix1 = confusion_matrix(y_test, y_pred, labels = SVM_poly1.classes_)
print("\nThe confusion matrix is:")
print(cnf_matrix1)
ax = sns.heatmap(cnf_matrix1, annot=True, fmt='d' )
ax.xaxis.set_ticklabels(SVM_poly1.classes_)
ax.yaxis.set_ticklabels(SVM_poly1.classes_)
plt.title("SVM with Polynomial kernel with C = 10 and degree 4")
plt.savefig(path+'/rbf_poly_C_10.png',format = 'png')

print(classification_report(y_test,y_pred))



coef = MODEL.coef_
fiction_coef = coef[0]
non_fiction_coef = coef[1]
Young_adult_coef = coef[2]

top_fiction_index = list(np.argsort(fiction_coef,axis=0)[-top_features:])
top_fiction_words = [words[i] for i in top_fiction_index]
top_fiction_vals = [fiction_coef[i] for i in top_fiction_index]
plt.bar(  x=  np.arange(10)  , height=top_fiction_vals, color = '#740001')
plt.xticks(np.arange(0, (10)), top_fiction_words, rotation=60, ha="right")
plt.title("Fiction Features")

top_non_fiction_index = list(np.argsort(non_fiction_coef,axis=0)[-top_features:])
top_non_fiction_words = [words[i] for i in top_non_fiction_index]
top_non_fiction_vals = [non_fiction_coef[i] for i in top_non_fiction_index]
plt.bar(  x=  np.arange(10)  , height=top_non_fiction_vals, color = '#0E1A40')
plt.xticks(np.arange(0, (10)), top_non_fiction_words, rotation=60, ha="right")
plt.title("Non Fiction Features")

top_YA_index = list(np.argsort(Young_adult_coef,axis=0)[-top_features:])
top_YA_words = [words[i] for i in top_YA_index]
top_YA_vals = [Young_adult_coef[i] for i in top_YA_index]
plt.bar(  x=  np.arange(10)  , height=top_YA_vals, color = '#1A472A')
plt.xticks(np.arange(0, (10)), top_YA_words, rotation=60, ha="right")
plt.title("Young Adult Features")


