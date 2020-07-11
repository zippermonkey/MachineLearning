#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 22:31:24 2020

@author: zippermonkey
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('voice.csv')
# =============================================================================
# Visualization
# 打印所有特征的kde曲线

male = data.loc[data.label=='male']
female = data.loc[data.label=='female']

plt.subplots(4,5,figsize=(20,15))
for i in range(1,21): 
    print(i,data.columns[i-1])
    ax = plt.subplot(4,5,i)
    ax.set_title(data.columns[i-1])
    sns.kdeplot(female.loc[female['label'] == 'female', female.columns[i-1]], shade=True,shade_lowest=False, label='F',ax =ax)
    sns.kdeplot(male.loc[male['label'] == 'male', male.columns[i-1]], shade=True,shade_lowest=False,label='M',ax = ax)

sns.pairplot(data[['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']],hue='label', height=3)
# =============================================================================

def prediction(data, model):
    y = np.array(data['label'])
    X = np.array(data.drop(['label'],1))
    
    # 归一化
    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # label 编码
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    
    # 数据集分割
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 30)
    
    
    model.fit(X_train,y_train)
    
    # 预测
    y_pred =model.predict(X_test)
    # 预测报告
    print(classification_report(y_test,y_pred,digits=4))
    
    # 交叉验证得分
    scores = cross_val_score(model,X,y,cv=5)
    print('cross_val_score: ')
    print(scores)
    print(scores.mean())
    return scores.mean()


# 特征的相关系数热图
f,ax = plt.subplots(figsize=(25, 15))
sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

# 特征选取  
"""
1. 首先可以从特征热图观察出特征区分比较明显的变量
2. 从相关系数中剔除相关程度很高的变量
3. 使用feature.selection包选取变量   观察不同选取数目的影响
"""
from sklearn.feature_selection import SelectKBest, f_classif

def select_kbest(data_frame,target,k=5):
    feat_selector = SelectKBest(f_classif, k=k)
    _ = feat_selector.fit(data_frame.drop(target, axis=1), data_frame[target])
    
    feat_scores = pd.DataFrame()
    feat_scores["F Score"] = feat_selector.scores_
    feat_scores["P Value"] = feat_selector.pvalues_
    feat_scores["Support"] = feat_selector.get_support()
    feat_scores["Attribute"] = data_frame.drop(target, axis=1).columns
    
    return feat_scores

k = 5
selected = select_kbest(data,'label',k = k).sort_values(['F Score'],ascending=False)
print(selected)


# print feture score 
plt.subplots(figsize=(25, 15))
k1=sns.barplot(x=selected['F Score'],y=selected['Attribute'])
k1.set_title('Feature Importance')

model = GaussianNB()
prediction(data,model)
a = []
b = []

for k in range(1,11):
    kbest = list(selected['Attribute'][:k])
    print('#####################################')
    print(k,':  ',kbest)
    print('#####################################')
    kbest.append('label')
    new_data = data[kbest]
    scores=prediction(new_data, model)
    a.append(k)
    b.append(scores)
   
plt.figure(figsize=(25,15))
plt.plot(a,b,marker = "o",markersize=10)
plt.xlabel('k')
plt.ylabel('accuracy')
plt.grid()
plt.show()

new_index = list(selected['Attribute'])
new_index.append('label')
new_data = data[new_index]
male = new_data.loc[new_data.label=='male']
female = new_data.loc[new_data.label=='female']

plt.subplots(4,5,figsize=(20,15))
for i in range(1,21): 
    ax = plt.subplot(4,5,i)
    ax.set_title(data.columns[i-1])
    sns.kdeplot(female.loc[female['label'] == 'female', female.columns[i-1]], shade=True,shade_lowest=False, label='F',ax =ax)
    sns.kdeplot(male.loc[male['label'] == 'male', male.columns[i-1]], shade=True,shade_lowest=False,label='M',ax = ax)
