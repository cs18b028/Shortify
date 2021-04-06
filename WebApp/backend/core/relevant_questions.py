import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from .process_data import process_text


data = pd.read_csv('../../data/processed_data_model1.csv')

w2v_model = Word2Vec.load('../../models/related_questions_model.bin')
words = w2v_model.wv.vocab.keys()

#word to numerical vector using the trained word embeddings

def question_to_vec(question, embeddings, dim = 300):
    question_embedding = np.zeros(dim) #initialize with zeros and dim = 300
    valid_words = 0
    for word in str(question).split(' '):
        if embeddings.wv.__contains__(word):
            valid_words+=1
            question_embedding += embeddings.wv.__getitem__(word)
        if valid_words>0:
            return question_embedding/valid_words
        else:
            return question_embedding


def rel_que(query):

    results = []

    # retrieving the save title embeddings

    em1 = pd.read_csv('../../models/title_embeddings1.csv')
    em2 = pd.read_csv('../../models/title_embeddings2.csv')
    em3 = pd.read_csv('../../models/title_embeddings3.csv')
    em4 = pd.read_csv('../../models/title_embeddings4.csv')

    em = em1.append([em2, em3, em4])

    title_embeddings = np.array(em)

    # process the query

    processed_query = process_text(query)

    results_returned = 1000 # number of results to be returned

    query_vect = np.array([question_to_vec(processed_query, w2v_model)]) # Vectorize the user query

    cosine_similarities = pd.Series(cosine_similarity(query_vect, title_embeddings)[0]) # calculate the cosine similarities

    relevant_questions = []
    max_cosine_score = max(cosine_similarities)
    if max_cosine_score==0:
        max_cosine_score = 1

    cos_weight = 10
    freq_weight = 80
    ans_weight = 6
    polar_weight = 2
    subj_weight = 2

    #score calculation

    for index, cosine_score in cosine_similarities.nlargest(results_returned).iteritems():
        
        freq_score = 0
        word_count = 0
        score = 0
        
        # modify for number of unique words

        for word in str(data.title[index].lower()).split():
            if word.lower() in processed_query:
                freq_score+=1
            word_count+=1
            
        freq_score/=word_count
        
        score =  (cos_weight*(cosine_score/max_cosine_score)+freq_weight*freq_score+ans_weight*data.score[index]+polar_weight*data.polarity[index]+subj_weight*data.subjectivity[index])
            
        relevant_questions.append((data.id[index], data.title[index], score))
        
    relevant_questions.sort(key = lambda x : x[2], reverse = True)

    for index, title, score in relevant_questions:
        results.append({
            'index' : str(index),
            'title' : title,
            'similarity_score' : str(score)
        })

    return results[0:1000]

def get_rel_que(query):
    results = rel_que(query)
    return results