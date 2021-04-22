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

stop_words = stopwords.words('english')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# %%
#rel_que = pd.read_csv('../../../data/relevant_questions.csv')
ans = pd.read_csv('../../../data/keyword_answer.csv')
kdf = pd.read_csv('../../../data/keywords.csv')


# %%
df1 = pd.read_csv('../../../data/original_data1.csv')
df2 = pd.read_csv('../../../data/original_data2.csv')
df3 = pd.read_csv('../../../data/original_data3.csv')
df4 = pd.read_csv('../../../data/original_data4.csv')
df5 = pd.read_csv('../../../data/original_data5.csv')
topics = ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6', 'Topic 7']


# %%
def length_text(text):
    if(type(text) != type(0.0)):
        text = text.split(' ')
        return len(text)
    else:
        return 0

# %%
def extract_text(text):
    soup = BeautifulSoup(text, 'lxml')
    txt = "".join([txt.text for txt in soup.find_all("p")])
    return txt

# %%
def get_dataframe(topic_list):
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

# %%
def cosine_score(question, answer):
    sentences = [question, answer]
    vectorizer = CountVectorizer()
    vector = vectorizer.fit_transform(sentences)
    text1 = vector.toarray()[0].tolist()
    text2 = vector.toarray()[1].tolist()
    cosc = 1-distance.cosine(text1, text2)
    return cosc
    
def entropy(text, tfidf_dict):
    if(type(text) == type(0.0)):
        return 0
    else:
        token = word_tokenize(text)
        entropy = 0.0
        for word in token:
            try:
                entropy = entropy + tfidf_dict[word]
            except:
                entropy = entropy + 0.0
        return entropy

# %%
def text_process(text):
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
# %%
def score(user_score, cosine_score, entropy_score, min_user_score, max_user_score, min_cosine_score, max_cosine_score, min_entropy_score, max_entropy_score):
    if(min_user_score == max_user_score):
        user_score = user_score - min_user_score
    else:
        user_score = ((user_score - min_user_score)/(max_user_score - min_user_score))
        
    if(min_cosine_score == max_cosine_score):
        cosine_score = cosine_score - min_cosine_score
    else:
        cosine_score = ((cosine_score - min_cosine_score)/(max_cosine_score - min_cosine_score))
        
    if(min_entropy_score == max_entropy_score):
        entropy_score = entropy_score - min_entropy_score
    else:
        entropy_score = ((entropy_score - min_entropy_score)/(max_entropy_score - min_entropy_score))
    return (user_score + cosine_score + entropy_score)

# %%
def ranking(query):
    query = text_process(query)
    rel_que = get_rel_que(query)

    ans['ans length'] = ans['answer'].apply(length_text)

    df = pd.concat([df1,df2,df3,df4,df5], axis=0)
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['answers', 'tags'], axis=1, inplace = True)

    rel_ans = pd.DataFrame()
    for i in enumerate(rel_que['id']):
        rel_ans = pd.concat([ans[ans['id'] == i[1]], rel_ans])

    rel_ans[topics] = rel_ans['topic list'].apply(get_dataframe)
    rel_ans.drop(columns=['topic list', 'id'], inplace = True)

    que_ans = pd.DataFrame()
    for i in enumerate(rel_que['id']):
        que_ans = pd.concat([df[df['id'] == i[1]], que_ans])

    que_ans['body'] = que_ans['body'].apply(extract_text)
    que_ans['question'] = que_ans['title'] + que_ans['body']
    que_ans.drop(columns=['title', 'body'], axis=1, inplace = True)

    que_ans['question list'] = que_ans['question'].apply(text_process)
    rel_ans['answer list'] = rel_ans['answer'].apply(text_process)

    data = rel_ans.join(que_ans)

    docs = data['answer list'].dropna().tolist()
    cv = CountVectorizer(stop_words = stop_words)
    word_count_vector = cv.fit_transform(docs)

    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit_transform(word_count_vector)

    tfidf_array = tfidf_transformer.idf_
    words = cv.get_feature_names()
    tfidf_dict = {}
    for i in range(0, len(tfidf_transformer.idf_)):
        tfidf_dict[words[i]] = tfidf_array[i]

    topics_data = []

    for topic in topics:
        topic_data = data[data[topic]==1][['id', 'answer', 'link', 'code', 'score', 'answer list', 'question list']]
        topic_df = pd.DataFrame()
        for i in range(0, topic_data.shape[0]):
            cosc = cosine_score(topic_data.iloc[i]['question list'], topic_data.iloc[i]['answer list'])
            entro = entropy(topic_data.iloc[i]['answer list'], tfidf_dict)
            topic_df = topic_df.append(pd.Series([topic_data.iloc[i]['answer'], topic_data.iloc[i]['link'], topic_data.iloc[i]['code'], topic_data.iloc[i]['score'], cosc, entro]), ignore_index=True)
        topic_df.columns = ['answer', 'link', 'code', 'score', 'cosine score', 'entropy']
        min_user_score = topic_df['score'].min()
        max_user_score = topic_df['score'].max()
        min_cosine_score = topic_df['cosine score'].min()
        max_cosine_score = topic_df['cosine score'].max()
        min_entropy_score = topic_df['entropy'].min()
        max_entropy_score = topic_df['entropy'].max()
        scores = []
        for i in range(0, topic_df.shape[0]):
            scores.append(score(topic_df.iloc[i]['score'], topic_df.iloc[i]['cosine score'], topic_df.iloc[i]['entropy'], min_user_score, max_user_score, min_cosine_score, max_cosine_score, min_entropy_score, max_entropy_score))
        topic_df.drop(columns = ['score', 'cosine score', 'entropy'], inplace=True)
        topic_df['sc'] = scores
        topic_df.sort_values(by = 'sc', ascending=False, inplace=True, kind='quicksort')
        topic_df = topic_df.head(15)
        topics_data.append(topic_df)
        #path = 'data/'+ topic +'.csv'
        #topic_df.to_csv(path, index=False)
    return topics_data





# %%
