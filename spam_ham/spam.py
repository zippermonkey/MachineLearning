# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd


# %%
data = pd.read_csv('spam_ham_dataset.csv')
data = data.iloc[:,2:]
# ham/0  spam/1


# %%
data.head(5)


# %%
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.figure(figsize=(6,4))
data['label_num'].value_counts().plot(kind='bar')


# %%
data['text'] = data['text'].str.lower()
data.head()


# %%
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer


# %%
stop_words=set(stopwords.words('english'))


# %%
stop_words.add('subject')


# %%
def text_process(text):
    tokenizer = RegexpTokenizer('[a-z]+')
    token = tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    token = [lemmatizer.lemmatize(w) for w in token if lemmatizer.lemmatize(w) not in stop_words]
    return token


# %%
data['text'] = data['text'].apply(text_process)


# %%
data.info()


# %%
data.head()


# %%
X = data['text']
y = data['label_num']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,train_size = 0.7)


# %%
train = pd.concat([X_train,y_train],axis=1)
test = pd.concat([X_test,y_test],axis = 1)


# %%
ham_train = train[train['label_num'] == 0] # 正常邮件
spam_train = train[train['label_num'] == 1] # 垃圾邮件


# %%
# 各取30封组成词库
ham_train_part = ham_train['text'].sample(30, random_state=42) 
spam_train_part = spam_train['text'].sample(30, random_state=42) 


# %%
part_words = []
for text in pd.concat([ham_train_part,spam_train_part]):
    part_words += text


# %%
part_words_set = set(part_words)
print(len(part_words_set))


# %%
import numpy as np
# 将正常邮件与垃圾邮件的单词都整理为句子，单词间以空格相隔，CountVectorizer()的句子里，单词是以空格分隔的
train_part_texts = [' '.join(text) for text in np.concatenate((spam_train_part.values, ham_train_part.values))]
# 训练集所有的单词整理成句子
train_all_texts = [' '.join(text) for text in train['text']]
# 测试集所有的单词整理成句子
test_all_texts = [' '.join(text) for text in test['text']]


# %%
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
cv = CountVectorizer()
part_fit = cv.fit(train_part_texts) # 以部分句子为参考
train_all_count = cv.transform(train_all_texts) # 对训练集所有邮件统计单词个数
test_all_count = cv.transform(test_all_texts) # 对测试集所有邮件统计单词个数
tfidf = TfidfTransformer()
train_tfidf_matrix = tfidf.fit_transform(train_all_count)
test_tfidf_matrix = tfidf.fit_transform(test_all_count)


# %%
from sklearn.naive_bayes import MultinomialNB
model  = MultinomialNB()
model.fit(train_tfidf_matrix, y_train)
model.score(test_tfidf_matrix,y_test)


# %%


