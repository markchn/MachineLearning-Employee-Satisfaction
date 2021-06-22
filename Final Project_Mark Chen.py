#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: markchen
"""

#Dataset link: https://drive.google.com/open?id=10bFq933mnmcDG9tL03t6TtLa-0ICkDMK

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

df_reviews = pd.read_csv('employee_reviews.csv')
df_reviews.sample(5)

print(df_reviews)


grouped_df_reviews = df_reviews.groupby('company').size()

print(grouped_df_reviews)



#Question 1
#Which company has the best overall-rating score? (To find the highest mean among 6 companies)

df_overall = df_reviews[['company','overall-ratings']].groupby('company').mean().sort_values('overall-ratings', ascending=False)
df_overall = df_overall.reset_index()

def plot_ratings():
    plt.figure(figsize=(10, 6))
    plt.ylim((3, 4.6))
    plt.bar(df_overall['company'],df_overall['overall-ratings'],width=0.45)
    plt.title('The average overall-rating score of each company')
    
    for x, y in zip(df_overall['company'], df_overall['overall-ratings']):
        plt.text(x, y , '%.2f' % y, ha='center', va='bottom')
        
plot_ratings()


f, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
sns.despine(left=True)
sns.set(style="white", palette="muted", color_codes=True)

facebook = df_reviews[(df_reviews.company == 'facebook')]['overall-ratings']
sns.distplot(facebook, color="g",ax=axes[0])

netflix = df_reviews[(df_reviews.company == 'netflix')]['overall-ratings']
sns.distplot(netflix, color="m",ax=axes[1])

plt.setp(axes, yticks=[])
plt.tight_layout()



#Qustion 2
#What’s the relationship between overall-rating score and these five stars? 
#(Five stars:  work-balance-stars, culture-value-stars, career-opportunities-stars, 
#comp-benefit-stars and senior-management-stars.)

#I need to do some data cleaning first in order to exclude reviews
# without those rating scores. I might use pairplots to help me 
# analyze the question. Then, I’ll use ordinary least squares regression
# to get more accurate regression results. I’m going to use seaborn and statsmodels.

print(df_reviews.columns)
#Index(['Unnamed: 0', 'company', 'location', 'dates', 'job-title', 'summary',
#       'pros', 'cons', 'advice-to-mgmt', 'overall-ratings',
#       'work-balance-stars', 'culture-values-stars',
#       'carrer-opportunities-stars', 'comp-benefit-stars',
#       'senior-mangemnet-stars', 'helpful-count', 'link'],
#      dtype='object')

df_reviews_heatmap=df_reviews[['overall-ratings','work-balance-stars','culture-values-stars',
                        'carrer-opportunities-stars','comp-benefit-stars',
                        'senior-mangemnet-stars']]

print(df_reviews_heatmap.shape)
#(67529, 6)

#Drop rows where ratings are none.
df_reviews_heatmap = df_reviews_heatmap.mask(df_reviews_heatmap.eq('none')).dropna()

#Converting strings to floats 
df_reviews_heatmap = df_reviews_heatmap.apply(pd.to_numeric, errors='ignore')

f, ax = plt.subplots(1,1,figsize=(9, 6))
sns.heatmap(df_reviews_heatmap[['overall-ratings','work-balance-stars','culture-values-stars',
                        'carrer-opportunities-stars','comp-benefit-stars',
                        'senior-mangemnet-stars']].corr(method='pearson'),annot = True,fmt='.3g')

ax = sns.heatmap(df_reviews_heatmap[['overall-ratings','work-balance-stars','culture-values-stars',
                        'carrer-opportunities-stars','comp-benefit-stars',
                        'senior-mangemnet-stars']].corr(method='pearson'),annot = True,fmt='.3g')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

df_reviews_heatmap.columns=['overall_ratings','work_balance_stars','culture_values_stars',
                            'carrer_opportunities_stars','comp_benefit_stars','senior_mangemnet_stars']

model1 = smf.ols('overall_ratings ~ work_balance_stars + culture_values_stars + carrer_opportunities_stars  + comp_benefit_stars + senior_mangemnet_stars'
                 , data=df_reviews_heatmap).fit()
print(model1.summary())



#Question 3
#What are the most frequently mentioned Pros and Cons of each company?

import re
import os
from spacy.lang.en.stop_words import STOP_WORDS

companys = ['facebook','google','apple','netflix','amazon','microsoft']

facebook= df_reviews[(df_reviews.company == 'facebook')][['pros','cons','advice-to-mgmt']]
facebook = facebook.mask(facebook.eq('none')).dropna().reset_index()

google= df_reviews[(df_reviews.company == 'google')][['pros','cons','advice-to-mgmt']]
google = google.mask(google.eq('none')).dropna().reset_index()

apple= df_reviews[(df_reviews.company == 'apple')][['pros','cons','advice-to-mgmt']]
apple = apple.mask(apple.eq('none')).dropna().reset_index()

netflix= df_reviews[(df_reviews.company == 'netflix')][['pros','cons','advice-to-mgmt']]
netflix = netflix.mask(netflix.eq('none')).dropna().reset_index()

amazon= df_reviews[(df_reviews.company == 'amazon')][['pros','cons','advice-to-mgmt']]
amazon= amazon.mask(amazon.eq('none')).dropna().reset_index()

microsoft= df_reviews[(df_reviews.company == 'microsoft')][['pros','cons','advice-to-mgmt']]
microsoft = microsoft.mask(microsoft.eq('none')).dropna().reset_index()

company_comments=[]
for j in range(len(companys)):
    comments = []
    pros = []
    cons = []
    advice = []
    tmp = eval('%s'%companys[j])
    for i in range(len(tmp)):
        pros.append(tmp['pros'][i])
        cons.append(tmp['cons'][i])
        advice.append(tmp['advice-to-mgmt'][i])
    comments.append(pros)
    comments.append(cons)
    comments.append(advice)
    company_comments.append(comments)

otherwords = ['facebook','google','apple','netflix','amazon','microsoft','company']
for word in otherwords:
    STOP_WORDS.add(word)

#Remove punctuation and any other non-alphabet characters
#Remove stopwords
df_nostop = []
temp = []
for i in range(len(companys)):
    for j in range(3):
        for k in range(len(company_comments[i][0])):
            company_comments[i][j][k] = company_comments[i][j][k].lower()
            company_comments[i][j][k] = re.sub(r'[^\w\s]', '', company_comments[i][j][k])
            company_comments[i][j][k] = company_comments[i][j][k].replace('_', "")
            company_comments[i][j][k] = re.sub(r'[0-9]+', " ",company_comments[i][j][k])
            for word in company_comments[i][j][k].split():
                if word not in STOP_WORDS:
                    temp.append(word)
            company_comments[i][j][k] = " ".join(temp)
            temp = []


import os
from os import path
from wordcloud import WordCloud

plt.subplots_adjust(left=3,right=5,bottom=2,top=5,hspace=0.1, wspace=0.1)

for i in range(6):
    plt.subplot(3,2,i+1)
    wordcloud = WordCloud().generate(" ".join(company_comments[i][0]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")


plt.subplots_adjust(left=3,right=5,bottom=2,top=5,hspace=0.1, wspace=0.1)

for i in range(6):
    plt.subplot(3,2,i+1)
    wordcloud = WordCloud().generate(" ".join(company_comments[i][1]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    

plt.subplots_adjust(left=3,right=5,bottom=2,top=5,hspace=0.1, wspace=0.1)

for i in range(6):
    plt.subplot(3,2,i+1)
    wordcloud = WordCloud().generate(" ".join(company_comments[i][2]))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")



#Question 4
#Can we use the Five Stars (the 5 other rating score mentioned in Question b) 
#to predict the overall rating score?

from sklearn.model_selection import train_test_split

df_reviews_predict=df_reviews[['overall-ratings','work-balance-stars','culture-values-stars',
                        'carrer-opportunities-stars','comp-benefit-stars',
                        'senior-mangemnet-stars']]

df_reviews_predict = df_reviews_predict.mask(df_reviews_predict.eq('none')).dropna()
df_reviews_predict = df_reviews_predict.apply(pd.to_numeric, errors='ignore')

df_train,df_test = train_test_split(df_reviews_predict, test_size=0.2)

print(len(df_train),len(df_test))


#Regression : Ordinary Least Squares Regression

from sklearn import linear_model
reg = linear_model.LinearRegression()
print(reg.fit(df_train.ix[:,1:6],df_train['overall-ratings']))

predicted_labels = reg.predict(df_test.ix[:,1:6])
df_test['predicted_reg'] = np.round(predicted_labels)

df_test_test = df_test.reset_index()
count = 0
for i in range(len(df_test_test)):
    if df_test_test['overall-ratings'][i]== df_test_test['predicted_reg'][i]:
        count +=1
print(count/len(df_test_test))


#Decision Trees

import sklearn.tree as sktree

dt_model = sktree.DecisionTreeClassifier(criterion='entropy')
# given first 4 columns, learn the species
dt_model.fit(df_train.ix[:,1:6],df_train['overall-ratings'])

# this is testing the model 
predicted_labels = dt_model.predict(df_test.ix[:,1:6])
df_test['predicted_label_tree'] = predicted_labels
print(dt_model.score(df_test.ix[:,1:6],df_test['overall-ratings']))


# Check feature importance
# Visualize the importances
feat_importance = dt_model.feature_importances_
pd.DataFrame({'Feature Importance':feat_importance},
            index=df_train.ix[:,1:6].columns).plot(kind='barh',figsize=(6,5))