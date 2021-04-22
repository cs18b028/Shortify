# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from numpy import nan
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None 
from sklearn.preprocessing import OneHotEncoder

from bs4 import BeautifulSoup

import nltk
import re
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial import distance

from .relevant_questions import get_rel_que

ans = pd.read_csv('../../../data/keyword_answer.csv')
kdf = pd.read_csv('../../../data/keywords.csv')

df1 = pd.read_csv('../../../data/original_data1.csv')
df2 = pd.read_csv('../../../data/original_data2.csv')
df3 = pd.read_csv('../../../data/original_data3.csv')
df4 = pd.read_csv('../../../data/original_data4.csv')
df5 = pd.read_csv('../../../data/original_data5.csv')

# %%
def normalised_score(score, min_score, max_score):
    score = ((score - min_score)/(max_score - min_score))
    return score

def length_text(text):
    if(type(text) != type(0.0)):
        text = text.split(' ')
        return len(text)
    else:
        return 0

def get_dataframe(topic_list):
    topics = ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6', 'Topic 7']
    if(type(topic_list) == type(0.0)):
        return pd.Series([0,0,0,0,0,0,0,0])
    else:
        topic_list = topic_list.split(', ')
        tl = []
        for topic in topics:
            if(topic in topic_list):
                val = 1
            else: 
                val = 0
            tl.append(val)
        return pd.Series(tl)

def extract_text(text):
    soup = BeautifulSoup(text, 'lxml')
    txt = "".join([txt.text for txt in soup.find_all("p")])
    return txt

def text_process(text):
    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    if(type(text) != type(0.0)):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r'[0-9]', ' ', text)
        text = text.strip()
        token = word_tokenize(text)
        text = [i for i in token if not i in stop_words]
        output = []
        for word in text:
            output.append(stemmer.stem(word))
        text = []
        for word in output:
            text.append(lemmatizer.lemmatize(word))
        text = " ".join(text)
    return text

def cosine_score(question, answer):
    sentences = [question, answer]
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(sentences)
    text1 = vector.toarray()[0].tolist()
    text2 = vector.toarray()[1].tolist()
    cosc = 1-distance.cosine(text1, text2)
    return cosc

def entropy(text, tfidf_array):
    if(type(text) == type(0.0)):
        return 0
    else:
        text = word_tokenize(text)
        entropy = 0
        for word in text:
            try:
                entropy = entropy + tfidf_array[word]
            except:
                entropy = entropy + 0
        return entropy

def ranking(query):
    #rel_que = pd.read_csv('data/relevant_questions.csv')
    query = ''
    rel_que = get_rel_que(query)
    # %%
    df = pd.concat([df1,df2,df3,df4,df5], axis=0)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['answers', 'tags'], axis=1, inplace = True)

    # %%
    ans['ans length'] = ans['answer'].apply(length_text)

    # %%
    rel_ans = pd.DataFrame()
    for i in enumerate(rel_que['id']):
        rel_ans = pd.concat([ans[ans['id'] == i[1]], rel_ans])

    # %%
    topics = ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6', 'Topic 7']
    # %%
    rel_ans[topics] = rel_ans['topic list'].apply(get_dataframe)
    rel_ans.drop(columns=['topic list', 'id'], inplace = True)
    # %%
    que_ans = pd.DataFrame()
    for i in enumerate(rel_que['id']):
        que_ans = pd.concat([df[df['id'] == i[1]], que_ans])
    # %%
    que_ans['body'] = que_ans['body'].apply(extract_text)
    que_ans['question'] = que_ans['title'] + que_ans['body']
    que_ans.drop(columns=['title', 'body'], axis=1, inplace = True)

    # %%
    stop_words = stopwords.words('english')
    # %%
    que_ans['question list'] = que_ans['question'].apply(text_process)
    rel_ans['answer list'] = rel_ans['answer'].apply(text_process)


    # %%
    data = rel_ans.join(que_ans)
    # %%
    #query = 'merge two lists in python'
    query = text_process(query)


    # %%
    docs = data['answer list'].dropna().tolist()
    cv = CountVectorizer(stop_words = stop_words)
    word_count_vector = cv.fit_transform(docs)

    # %%
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit_transform(word_count_vector)

    # %%
    tfidf_array = tfidf_transformer.idf_
    words = cv.get_feature_names()
    tfidf_dict = {}
    for i in range(0, len(tfidf_transformer.idf_)):
        tfidf_dict[tfidf_array[i]] = words[i]
    # %%
    topic_data_frame = []
    for topic in topics:
        topic_data = data[data[topic]==1][['id', 'answer', 'link', 'code', 'score', 'answer list', 'question list']]
        #entropy_score = []
        topic_df = pd.DataFrame()
        for i in range(0, topic_data.shape[0]):
            cosc = cosine_score(topic_data.iloc[i]['question list'], topic_data.iloc[i]['answer list'])
            topic_df = topic_df.append(pd.Series([topic_data.iloc[i]['answer'], topic_data.iloc[i]['link'], topic_data.iloc[i]['code'], topic_data.iloc[i]['score'], cosc]), ignore_index=True)
        topic_df.columns = ['answer', 'link', 'code', 'score', 'cosine score']
        for i in range(0, topic_data.shape[0]):
            topic_df.iloc[i]['entropy'] = entropy(data['answer list'], tfidf_array)
        min_user_score = topic_df['score'].min()
        max_user_score = topic_df['score'].max()
        min_cosine_score = topic_df['cosine score'].min()
        max_cosine_score = topic_df['cosine score'].max()
        min_entropy_score = topic_df['entropy'].min()
        max_entropy_score = topic_df['entropy'].max()
        for i in range(0, topic_df.shape[0]):
            topic_df.iloc[i]['score'] = normalised_score(topic_df.iloc[i]['score'], min_user_score, max_user_score)
            topic_df.iloc[i]['cosine score'] = normalised_score(topic_df.iloc[i]['cosine score'], min_cosine_score, max_cosine_score)
            topic_df.iloc[i]['entropy'] = normalised_score(topic_df.iloc[i]['entropy'], min_entropy_score, max_entropy_score)
        topic_df['sc'] = topic_df['score'] + topic_df['cosine score'] + topic_df['entropy']
        topic_df.sort_values(by = 'sc', ascending=False, inplace=True, kind='quicksort')
        topic_df.drop(columns = ['score', 'cosine score', 'entropy'], inplace=True)
        topic_data_frame.append(topic_df)
    return topic_data_frame
        #path = 'data/'+ topic +'.csv'
        #topic_df = topic_df.head(15)
        #topic_df.to_csv(path, index=False)
# %%



