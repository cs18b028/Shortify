######################################################################################################
#                                                                                                    #
#   Summarizer Module - Given a topic-wise list of ranked answers (from the ranking model) makes     #
#   summary outof them under each topic using MMR summarizer model                                   #
#                                                                                                    #
#   functions : processAns, TFs, IDFs, TF_IDF, sentenceSim, buildBase, bestSentence, MMRScore,       #
#   makeSummary, summarizer                                                                          #
#                                                                                                    #
#   classes : sentence                                                                               #
#                                                                                                    #
######################################################################################################

# Importing python libraries and modules

import os
import math
import string
import re
import nltk
import pandas as pd
from core.ranking import ranking

kdf = pd.read_csv('../../data/keywords.csv') # Getting the keywords under each topic

# sentence - class that is used to store the sentences for each answer and their related info

class sentence:
    
    def __init__(self, id, question, processedWords, originalWords):
        self.id = id
        self.question = question
        self.processedWords = processedWords
        self.originalWords = originalWords
        self.wordFreq = self.sentWordFreq()

    # Getter functions

    def getId(self):
        return self.id  # id to keep track of the question from which the answer is taken

    def getQue(self):
        return self.question  # question title from which the answer is taken

    def getProcessedWords(self):
        return self.processedWords # preprocessed words of the sentence

    def getOriginalWords(self):
        return self.originalWords # original words of the sentence
    
    def getWordFreq(self):
        return self.wordFreq  # dictionary of word frequencies

    def sentWordFreq(self): # function to calculate the word frequencies
        wordFreq = {}
        for word in self.processedWords:
            if word not in wordFreq.keys():
                wordFreq[word] = 1
            else:
                wordFreq[word] = wordFreq[word] + 1
        return wordFreq


# processAns - function to process an answer : tokenize, lowercase, punctuation removal and stemming and return sentence objects

def processAns(id, question, answer):

    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')

    lines = sentence_token.tokenize(answer.strip()) # Getting sentences from answer

    sentences = []  # A list to store the sentence objects for each sentence in the answer
    porter = nltk.PorterStemmer()

    for line in lines:

        original = nltk.word_tokenize(line[:])

        originalWords = ""

        stemmedSent = []    # Stores the list of the processed words in each sentence

        for word in original:
            new_word = re.sub(r'[^\w\s]', '', porter.stem(word.strip().lower())) # The processing step

            if new_word!='':
                origin_word = ""
                if new_word in query_words:
                    origin_word = "<b>" + word + "</b>" # Aesthetic related
                else:
                    origin_word = word
                originalWords = originalWords + " " + origin_word
                stemmedSent.append(new_word)
        
        if stemmedSent != [] :
            sentences.append(sentence(id, question, stemmedSent, originalWords)) # Placing sentence objects in the sentences list

    return sentences        


# TFs - function for TF score calculation

def TFs(sentences):
    
    tfs = {}
    
    for sent in sentences:
        wordFreqs = sent.getWordFreq()

        # TF for each word

        for word in wordFreqs.keys():
            if tfs.get(word, 0)!=0:
                tfs[word] = tfs[word] + wordFreqs[word]
            else:
                tfs[word] = wordFreqs[word]
        
    return tfs


# IDFs - function to calculate the IDF score

def IDFs(sentences):
    
    N = len(sentences)
    idf = 0
    idfs = {}
    words = {}
    w2 = []
    
    for sent in sentences:

        for word in sent.getProcessedWords():

            if sent.getWordFreq().get(word, 0) != 0:
                words[word] = words.get(word, 0)+1

    for word in words:
        n = words[word]

        try:
            w2.append(n)                
            idf = math.log10(float(N)/n)
        except ZeroDivisionError:
            idf = 0

        idfs[word] = idf    # IDF for each word

    return idfs


# TF_IDF - function to calculate the TF_IDF score for each word

def TF_IDF(sentences):
    tfs = TFs(sentences)
    idfs = IDFs(sentences)
    retval = {}

    for word in tfs:
        tf_idfs = tfs[word]*idfs[word] # The calulation step

        if retval.get(tf_idfs, None) == None:
            retval[tf_idfs] = [word]
        else:
            retval[tf_idfs].append(word)

    return retval


# sentenceSim - function to calculate the similarity between two sentences with the help of the IDF scores

def sentenceSim(sentence1, sentence2, IDF):
    
    numerator = 0
    denominator = 0

    # sentence2

    for word in sentence2.getProcessedWords():
        numerator+= sentence1.getWordFreq().get(word,0) * sentence2.getWordFreq().get(word,0) *  IDF.get(word,0) ** 2

    # sentence1

    for word in sentence1.getProcessedWords():
        denominator+= ( sentence1.getWordFreq().get(word,0) * IDF.get(word,0) ) ** 2

    # calculation of similarity

    try:
        return numerator / math.sqrt(denominator)
    except ZeroDivisionError:
        return float("-inf")

query_words = []

# buildBase - function to make a sentence that has all the processed query words and the highest TF_IDF scores

def buildBase(query, TF_IDF_dict, n):

    scores = TF_IDF_dict.keys()
    sorted(scores, reverse=True) # Sorting in the decreasing order og TF IDF scores

    i = 0
    j = 0
    baseWords = [] # List to carry the words

    # Adding processed query words

    for word in query.getProcessedWords():
        baseWords.append(word)
        i = i+1
        if(i>n):
            break

    # Adding the highest TF_IDF scored words

    while(i<n):
        words = TF_IDF_dict[list(scores)[j]]
        for word in words:
            baseWords.append(word)
            i = i+1
            if(i>n):
                break
        j = j+1

        return sentence("base", query.getQue(), baseWords, baseWords)


# bestSentence - function to get the sentence that has highest similarity with the base

def bestSentence(sentences, base, IDF):
    best_sentence = None
    maxVal = float("-inf")

    for sent in sentences:
        similarity = sentenceSim(sent, base, IDF)

        if similarity > maxVal:
            best_sentence = sent
            maxVal = similarity

    if best_sentence != None:
        sentences.remove(best_sentence)

    return best_sentence

# MMRScore - function to calculate the MMR score

def MMRScore(Si, base, Sj, lambta, IDF):
    Sim1 = sentenceSim(Si, base, IDF)
    l_expr = lambta * Sim1
    value = [float("-inf")]

    for sent in Sj:
        Sim2 = sentenceSim(Si, sent, IDF)
        value.append(Sim2)

    r_expr = (1-lambta)*max(value) # lambta is a hyper parameter
    MMR_SCORE = l_expr-r_expr

    return MMR_SCORE

# makeSummary - function that makes summary based on the MMRscore

def makeSummary(sentences, best_sentence, base, summary_len, lambta, IDF):
    summary = [best_sentence] # Start with best sentence
    sum_len = len(best_sentence.getProcessedWords()) 

    while(sentences != [] and sum_len<summary_len):
        MMRval={}

        # Assiging the MMR score for each sentence

        for sent in sentences:
            MMRval[sent] = MMRScore(sent, base, summary, lambta, IDF)
        
        maxxer = max(MMRval, key= lambda x: MMRval[x]) # Sorting the sentences in descending order of MMR score
        if maxxer != None:
            summary.append(maxxer)  # Get the sentence with max MMR score and put it in the summary
            sentences.remove(maxxer)
            sum_len += len(maxxer.getProcessedWords())
        
    return summary

# summarizer - function that serves as the entry point for this modules

def summarizer(query):

    topic_ans = ranking(query) # Getting the list of topic-wise ranked answers

    link = "https://stackoverflow.com/questions/"

    summaries = [] # List that stores the summaries under each topic

    topic_num = 0

    query_sent_list = processAns("query", query, query)

    try: 
        query_sent = query_sent_list[0] # User query sentence object

        global query_words

        query_words = query_sent.getProcessedWords() # Processed words from user query

        for topic in topic_ans:

            answers = topic[["id", "question", "answer"]].values.tolist()

            sentences = []

            # forming sentences from all the answers in each topic

            for answer in answers:
                sentences = sentences + processAns(answer[0], answer[1], answer[2])
                
            IDF = IDFs(sentences)
            TF_IDF_dict = TF_IDF(sentences) # Calculating the TF IDF scores for each word

            base = buildBase(query_sent, TF_IDF_dict, 10) # Making the list of base words (most preferable words)

            bestsent = bestSentence(sentences, base, IDF) # Best sentence

            summary = makeSummary(sentences, bestsent, base, 100, 0.5, IDF) # Summary

            # Filling in the summaries list with the topic related keywords and the summaries under each topic

            final_summary = ""
            for sent in summary:
                que = str(sent.getQue())
                final_summary = final_summary + "<a href"+ "=" + link + str(int(sent.getId())) + " target='_blank' title='"+ que +"'/>" + sent.getOriginalWords() + "</a>. "
            final_summary = final_summary[:-1]

            # Topics

            topic_keyword = kdf['Keywords'].iloc[topic_num]
            
            summaries.append({
                'topic': topic_keyword,
                'summary': final_summary
            })

            topic_num = topic_num + 1

    except:
        summaries.append({
            'topic': "",
            'summary': "<h5 align=\"center\">No results! Please try a different query</h5>"
        })

    return summaries