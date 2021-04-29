######################################################################################################
#                                                                                                    #
#   Ranking Module - Given a dataframe of relevant questions, their corresponding answers are ranked #
#   using few score metrics and the top 15 questions of each topic are returned as a list            #
#                                                                                                    #
#   functions :  get_dataframeIDFs, TF_IDF, sentenceSim, buildBase, bestSentence, MMRScore,          #
#   makeSummary, summarizer                                                                          #                                                                       #
#                                                                                                    #
######################################################################################################

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

from core.relevant_questions import rel_questions

stop_words = stopwords.words('english')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

#'rel_que' dataframe contains the questions that are relevant to the query which is the output of relevant questions model
rel_que = pd.read_csv('data/relevant_questions.csv')
#'ans' dataframe contains all the answers with their topics
ans = pd.read_csv('data/keyword_answer.csv')
#'kdf' dataframe contains the data of keywords for each topic
kdf = pd.read_csv('data/keywords.csv')


topics = ['Topic 0', 'Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5', 'Topic 6', 'Topic 7']


#calculates the length of text
def length_text(text):
    if(type(text) != type(0.0)):
        text = text.split(' ')
        return len(text)
    else:
        return 0


def extract_text(text):
    soup = BeautifulSoup(text, 'lxml')
    txt = "".join([txt.text for txt in soup.find_all("p")])
    return txt


#Gives one-hot encoding of the topics for the answers
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


#a function for similarity of the answers with their corresponding questions
def cosine_score(question, answer):
    if(type(answer) == type(0.0) or type(question) == type(0.0)):
        return 0
    else:
        sentences = [question, answer]
        vectorizer = CountVectorizer()
        vector = vectorizer.fit_transform(sentences)
        text1 = vector.toarray()[0].tolist()
        text2 = vector.toarray()[1].tolist()
        cosc = 1-distance.cosine(text1, text2)
        return cosc

#a function that caluclates the sum of tfidf of each word in a text
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

#a function that calculates similarity with the query
def query_sim_score(query, answer):
    score = 0
    if(type(answer) == type(0.0)):
        return 0
    else:
        query = query
        for q in query:
            if(q in answer):
                score =  score + 1
        score = score/len(query)
        return score


# a text processing function which converts text into lowercase, removes any characters like numbers, punctuation, 
# removes stopwords, stems the words and lemmatises them
def text_process(text):
    if(type(text) != type(0.0)):
        text = text.lower()                              
        text = re.sub("[^a-z]", " ", text)               
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

#normalises all the scores and returns their weighted sum
def score(user_score, cosine_score, entropy_score, sim_score, min_user_score, max_user_score, min_cosine_score, max_cosine_score, min_entropy_score, max_entropy_score):
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
    return (user_score + cosine_score + entropy_score + 2 * sim_score)

def ranking(query):
    rel_que = rel_questions(query)
    query = text_process(query)

    ans['ans length'] = ans['answer'].apply(length_text)

    #'rel_ans' dataframe contains the answers corresponding to the relevant questions to the query i.e, questions of 'rel_que' dataset
    rel_ans = pd.DataFrame()
    for i in enumerate(rel_que['id']):
        rel_ans = pd.concat([ans[ans['id'] == i[1]], rel_ans])

    rel_ans[topics] = rel_ans['topic list'].apply(get_dataframe)
    rel_ans.drop(columns=['topic list', 'id'], inplace = True)
    rel_ans['answer list'] = rel_ans['answer'] + rel_ans['code']

    rel_ans['title'] = rel_que['questions']
    rel_ans['question list'] = rel_ans['title'] + rel_ans['body']
    rel_ans.drop(columns=['body'], axis=1, inplace = True)

    #answers and questions are of 'rel_ans' dataframe are preprocessed
    rel_ans['question list'] = rel_ans['question list'].apply(text_process)
    rel_ans['answer list'] = rel_ans['answer list'].apply(text_process)

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
        topic_data.dropna()
        topic_df = pd.DataFrame()
        for i in range(0, topic_data.shape[0]):
            cosc = cosine_score(topic_data.iloc[i]['question list'], topic_data.iloc[i]['answer list'])
            entro = entropy(topic_data.iloc[i]['answer list'], tfidf_dict)
            sim_score = query_sim_score(query, topic_data.iloc[i]['answer list'])
            topic_df = topic_df.append(pd.Series([topic_data.iloc[i]['id'], topic_data.iloc[i]['answer'], topic_data.iloc[i]['link'], topic_data.iloc[i]['code'], topic_data.iloc[i]['score'], cosc, entro, sim_score]), ignore_index=True)
        if topic_df.shape[0]==0:
            topics_data.append(topic_df)
        else:
            topic_df.columns = ['id', 'answer', 'link', 'code', 'score', 'cosine score', 'entropy','sim score']
            min_user_score = topic_df['score'].min()
            max_user_score = topic_df['score'].max()
            min_cosine_score = topic_df['cosine score'].min()
            max_cosine_score = topic_df['cosine score'].max()
            min_entropy_score = topic_df['entropy'].min()
            max_entropy_score = topic_df['entropy'].max()
            scores = []
            for i in range(0, topic_df.shape[0]):
                scores.append(score(topic_df.iloc[i]['score'], topic_df.iloc[i]['cosine score'], topic_df.iloc[i]['entropy'], 
                                    topic_df.iloc[i]['sim score'], min_user_score, max_user_score, min_cosine_score, max_cosine_score,
                                    min_entropy_score, max_entropy_score))
            topic_df['sc'] = scores
            topic_df.sort_values(by = 'sc', ascending=False, inplace=True, kind='quicksort')
            topic_df.drop(columns = ['score', 'cosine score', 'entropy', 'sim score', 'sc'], inplace=True)
            topic_df = topic_df.head(15)
            topics_data.append(topic_df)
    return topics_data
