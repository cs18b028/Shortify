import os
import math
import string
import re
import nltk
from core.ranking import ranking

class sentence:
    
    def __init__(self, id, processedWords, originalWords):
        self.id = id
        self.processedWords = processedWords
        self.originalWords = originalWords
        self.wordFreq = self.sentWordFreq()

    def getId(self):
        return self.id

    def getProcessedWords(self):
        return self.processedWords

    def getOriginalWords(self):
        return self.originalWords
    
    def getWordFreq(self):
        return self.wordFreq

    def sentWordFreq(self):
        wordFreq = {}
        for word in self.processedWords:
            if word not in wordFreq.keys():
                wordFreq[word] = 1
            else:
                wordFreq[word] = wordFreq[word] + 1
        return wordFreq

query_words = []

def processAns(id, answer):

    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    lines = sentence_token.tokenize(answer.strip())

    sentences = []
    porter = nltk.PorterStemmer()

    for line in lines:

        original = nltk.word_tokenize(line[:])

        originalWords = ""

        stemmedSent = []

        for word in original:
            new_word = re.sub(r'[^\w\s]', '', porter.stem(word.strip().lower()))
            if new_word!='':
                origin_word = ""
                if new_word in query_words:
                    origin_word = "<b>" + word + "</b>"
                else:
                    origin_word = word
                originalWords = originalWords + " " + origin_word
                stemmedSent.append(new_word)
        
        if stemmedSent != [] :
            sentences.append(sentence(id, stemmedSent, originalWords))

    return sentences        


def TFs(sentences):
    
    tfs = {}
    
    for sent in sentences:
        wordFreqs = sent.getWordFreq()

        for word in wordFreqs.keys():
            if tfs.get(word, 0)!=0:
                tfs[word] = tfs[word] + wordFreqs[word]
            else:
                tfs[word] = wordFreqs[word]
        
    return tfs

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

        idfs[word] = idf

    return idfs

def TF_IDF(sentences):
    tfs = TFs(sentences)
    idfs = IDFs(sentences)
    retval = {}

    for word in tfs:
        tf_idfs = tfs[word]*idfs[word]

        if retval.get(tf_idfs, None) == None:
            retval[tf_idfs] = [word]
        else:
            retval[tf_idfs].append(word)

    return retval

def sentenceSim(sentence1, sentence2, IDF):
    
    numerator = 0
    denominator = 0

    for word in sentence2.getProcessedWords():
        numerator+= sentence1.getWordFreq().get(word,0) * sentence2.getWordFreq().get(word,0) *  IDF.get(word,0) ** 2

    for word in sentence1.getProcessedWords():
        denominator+= ( sentence1.getWordFreq().get(word,0) * IDF.get(word,0) ) ** 2

    try:
        return numerator / math.sqrt(denominator)
    except ZeroDivisionError:
        return float("-inf")

def buildBase(query, TF_IDF_dict, n):

    scores = TF_IDF_dict.keys()
    sorted(scores, reverse=True)

    i = 0
    j = 0
    baseWords = []

    for word in query.getProcessedWords():
        baseWords.append(word)
        i = i+1
        if(i>n):
            break

    while(i<n):
        words = TF_IDF_dict[list(scores)[j]]
        for word in words:
            baseWords.append(word)
            i = i+1
            if(i>n):
                break
        j = j+1

        return sentence("base", baseWords, baseWords)

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

def MMRScore(Si, base, Sj, lambta, IDF):
    Sim1 = sentenceSim(Si, base, IDF)
    l_expr = lambta * Sim1
    value = [float("-inf")]

    for sent in Sj:
        Sim2 = sentenceSim(Si, sent, IDF)
        value.append(Sim2)

    r_expr = (1-lambta)*max(value)
    MMR_SCORE = l_expr-r_expr

    return MMR_SCORE

def makeSummary(sentences, best_sentence, base, summary_len, lambta, IDF):
    summary = [best_sentence]
    sum_len = len(best_sentence.getProcessedWords())

    while(sentences != [] and sum_len<summary_len):
        MMRval={}

        for sent in sentences:
            MMRval[sent] = MMRScore(sent, base, summary, lambta, IDF)
        
        maxxer = max(MMRval, key= lambda x: MMRval[x])
        if maxxer != None:
            summary.append(maxxer)
            sentences.remove(maxxer)
            sum_len += len(maxxer.getProcessedWords())
        
    return summary

def summarizer(query):

    topic_ans = ranking(query)

    link = "https://stackoverflow.com/questions/"

    summaries = []

    query_sent_list = processAns("query", query)
    query_sent = query_sent_list[0]

    global query_words

    query_words = query_sent.getProcessedWords()

    print(query_words)

    for topic in topic_ans:

        answers = topic[["id", "answer"]].values.tolist()

        sentences = []

        for answer in answers:
            sentences = sentences + processAns(answer[0], answer[1])
            
        IDF = IDFs(sentences)
        TF_IDF_dict = TF_IDF(sentences)

        base = buildBase(query_sent, TF_IDF_dict, 10)

        bestsent = bestSentence(sentences, base, IDF)

        summary = makeSummary(sentences, bestsent, base, 100, 0.5, IDF)

        final_summary = ""
        for sent in summary:
            final_summary = final_summary + "<a href"+ "=" + link + str(int(sent.getId())) + " target='_blank'>" + sent.getOriginalWords() + " </a> "
        final_summary = final_summary[:-1]
        
        summaries.append({
            'topic': 'python, lists, merge sort',
            'summary': final_summary
        })

    return summaries