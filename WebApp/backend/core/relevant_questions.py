######################################################################################################
#                                                                                                    #
#   Relevant Questions Retrievel Module - takes a user query gets the questions most relevant to it  #
#   from the stackoverflow dataset previously retrieved and processed                                #
#                                                                                                    #
#   functions : question_to_vec, rel_questions, get_rel_que                                          #
#                                                                                                    #
######################################################################################################

# import python libraries and modules

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from core.process_data import process_text

# Loading processed dataset for this model

data = pd.read_csv('../../data/processed_data_model1.csv')

# Importing the trained and saved word2vec model from the models folder

w2v_model = Word2Vec.load('../../models/related_questions_model.bin')
words = w2v_model.wv.vocab.keys()


# question_to_vec - Function to convert a word to a numerical vector using the trained word embeddings

def question_to_vec(question, embeddings, dim = 300):

    # initialize with zeros and dim = 300

    question_embedding = np.zeros(dim)
    valid_words = 0

    for word in str(question).split(' '):
        if embeddings.wv.__contains__(word):
            valid_words+=1
            question_embedding += embeddings.wv.__getitem__(word)
        if valid_words>0:
            return question_embedding/valid_words
        else:
            return question_embedding


# rel_questions - function to get relevant questions, uses cosine similarity to measure the similarity between
# the query and questions in the dataset (in their numerical vector form obtained using question_to_vec function)
# also uses word frequency, answer score, polarity and subjectivity factors to measure similarity

def rel_questions(query):

    # results - this list stores the final results and is given to the questions endpoint in app.py

    global results

    results = []

    # Retrieving the save title embeddings

    em1 = pd.read_csv('../../models/title_embeddings1.csv')
    em2 = pd.read_csv('../../models/title_embeddings2.csv')
    em3 = pd.read_csv('../../models/title_embeddings3.csv')
    em4 = pd.read_csv('../../models/title_embeddings4.csv')

    em = em1.append([em2, em3, em4])

    title_embeddings = np.array(em)

    # Processing the query

    processed_query = process_text(query)

    results_returned = 100 # Number of relevant questions to be returned

    query_vect = np.array([question_to_vec(processed_query, w2v_model)]) # Vectorize the user query

    cosine_similarities = pd.Series(cosine_similarity(query_vect, title_embeddings)[0]) # Calculate the cosine similarities

    relevant_questions = []
    max_cosine_score = max(max(cosine_similarities), 0.0001)

    # Weights for various factors in the similarity score

    cos_weight = 30
    freq_weight = 60
    ans_weight = 6
    polar_weight = 2
    subj_weight = 2

    link = "https://stackoverflow.com/questions/"

    # Similarity score calculation

    for index, cosine_score in cosine_similarities.nlargest(results_returned).iteritems():
        
        freq_score = 0
        word_count = 0
        score = 0
        
        # Word frequency calculation - high value indicates more query related words in a question

        for word in data.processed_title[index].split():
            if word.lower() in processed_query:
                freq_score+=1
            word_count+=1
            
        freq_score/=word_count

        # Similarity score involving cosine similarity, word frequency, answer score, polarity and subjectivity
        
        score = (cos_weight*(cosine_score/max_cosine_score)+freq_weight*freq_score+ans_weight*data.score[index]+polar_weight*data.polarity[index]+subj_weight*data.subjectivity[index])/100
            
        relevant_questions.append((index, data.id[index], data.title[index], score))
        
    relevant_questions.sort(key = lambda x : x[3], reverse = True)

    # Putting the obtained relevant questions into the results array

    for index, id, question, score in relevant_questions:
        results.append({
            'question': "<a href"+ "=" + link + str(int(id)) + " target='_blank'>" + question + "</a>",
            'score': str(score*10)
        })

    df = pd.DataFrame(relevant_questions).iloc[:,1:]
    df.columns = ['id','questions','score'] # Dataframe with question id, questions and score

    return df


# get_rel_que - function that return questions for the questions endpoint in the app.py

def get_rel_que(query):
    rel_questions(query)
    return results[0:20] # Returning top 20 relevant questions