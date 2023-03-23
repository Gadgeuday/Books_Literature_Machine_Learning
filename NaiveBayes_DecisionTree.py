# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:54:10 2023

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

vectorizer = CountVectorizer(stop_words = 'english',max_df=0.8)
X = vectorizer.fit_transform(NYT['description'])

words = vectorizer.get_feature_names()

X = X.toarray()
X = preprocessing.normalize(X)

y = NYT['Genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


from sklearn.naive_bayes import MultinomialNB
NB= MultinomialNB()

NB.fit(X_train, y_train)

ypred = NB.predict(X_test)

from sklearn.metrics import confusion_matrix

cnf_matrix1 = confusion_matrix(y_test, ypred, labels = NB.classes_)
print("\nThe confusion matrix is:")
print(cnf_matrix1)
ax = sns.heatmap(cnf_matrix1, annot=True, fmt='d' )
ax.xaxis.set_ticklabels(NB.classes_)
ax.yaxis.set_ticklabels(NB.classes_)

print(classification_report(y_test,ypred))

NB_priors = pd.DataFrame(np.exp(NB.feature_log_prob_).T,columns = ['Fiction','Non Fiction','Young Adult'])
NB_priors['Word'] = words
NB_priors['Tot_prob'] = NB_priors['Fiction']+NB_priors['Non Fiction']+NB_priors['Young Adult']
NB_priors_top = NB_priors.sort_values(by = 'Tot_prob',ascending = False).head(15)

NB_priors_top.drop('Tot_prob',axis=1,inplace=True)

NB_priors_top.index = NB_priors_top['Word']

NB_priors_top.plot(kind='barh', stacked=True, color=['#740001', '#0E1A40', '#1A472A'])
plt.xticks([])
plt.ylabel('Word')
plt.xlabel('Feature Probability')



plt.savefig(path+"/NB_Feature_priors.png")



### DT 1 with entropy and min samples to be 10

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report

# Decision Tree 1

DT=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=20,
                            min_samples_leaf=10)
DT.fit(X_train, y_train)

ypred = DT.predict(X_test)

from sklearn.metrics import confusion_matrix

cnf_matrix2 = confusion_matrix(y_test, ypred)
print("\nThe confusion matrix is:")
print(cnf_matrix2)
ax = sns.heatmap(cnf_matrix2, annot=True, fmt='d' )
ax.xaxis.set_ticklabels(DT.classes_)
ax.yaxis.set_ticklabels(DT.classes_)

print(classification_report(y_test,ypred))

tree.plot_tree(DT,
               feature_names = words, 
               class_names=['Fiction','Non Fiction','Young Adult'],
               filled = True)

plt.savefig(path+"/DT1.pdf")


# Decision Tree 2

DT=DecisionTreeClassifier(criterion='gini', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=20,
                            min_samples_leaf=10)
DT.fit(X_train, y_train)

ypred = DT.predict(X_test)

from sklearn.metrics import confusion_matrix

cnf_matrix2 = confusion_matrix(y_test, ypred)
print("\nThe confusion matrix is:")
print(cnf_matrix2)
ax = sns.heatmap(cnf_matrix2, annot=True, fmt='d' )
ax.xaxis.set_ticklabels(DT.classes_)
ax.yaxis.set_ticklabels(DT.classes_)

print(classification_report(y_test,ypred))

tree.plot_tree(DT,
               feature_names = words, 
               class_names=['Fiction','Non Fiction','Young Adult'],
               filled = True)

plt.savefig(path+"/DT2.pdf")

# Decision Tree 3

DT=DecisionTreeClassifier(criterion='gini', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=10,
                            min_samples_leaf=10)
DT.fit(X_train, y_train)

ypred = DT.predict(X_test)

from sklearn.metrics import confusion_matrix

cnf_matrix2 = confusion_matrix(y_test, ypred)
print("\nThe confusion matrix is:")
print(cnf_matrix2)
ax = sns.heatmap(cnf_matrix2, annot=True, fmt='d' )
ax.xaxis.set_ticklabels(DT.classes_)
ax.yaxis.set_ticklabels(DT.classes_)

print(classification_report(y_test,ypred))

tree.plot_tree(DT,
               feature_names = words, 
               class_names=['Fiction','Non Fiction','Young Adult'],
               filled = True)

plt.savefig(path+"/DT3.pdf")

# Decision Tree 4

DT=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                           # max_depth=20,
                            min_samples_leaf=10)
DT.fit(X_train, y_train)

ypred = DT.predict(X_test)

from sklearn.metrics import confusion_matrix

cnf_matrix2 = confusion_matrix(y_test, ypred)
print("\nThe confusion matrix is:")
print(cnf_matrix2)
ax = sns.heatmap(cnf_matrix2, annot=True, fmt='d' )
ax.xaxis.set_ticklabels(DT.classes_)
ax.yaxis.set_ticklabels(DT.classes_)

print(classification_report(y_test,ypred))

tree.plot_tree(DT,
               feature_names = words, 
               class_names=['Fiction','Non Fiction','Young Adult'],
               filled = True)

plt.savefig(path+"/DT1.pdf")
