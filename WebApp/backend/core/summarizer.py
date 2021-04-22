import os
import math
import string
import re
import nltk

from IPython.display import HTML

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

def processAns(id, answer):

    sentence_token = nltk.data.load('tokenizers/punkt/english.pickle')
    lines = sentence_token.tokenize(answer.strip())

    sentences = []
    porter = nltk.PorterStemmer()

    for line in lines:
        originalWords = line[:]
        line = line.strip().lower()

        sent = nltk.word_tokenize(line)

        stemmedSent = [re.sub(r'[^\w\s]', '', porter.stem(word)) for word in sent]

        new_stemsent = []

        for word in stemmedSent:
            if word!='':
                new_stemsent.append(word)

        stemmedSent = new_stemsent
        
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

def buildBase(sentences, TF_IDF_dict, n):

    scores = TF_IDF_dict.keys()
    sorted(scores, reverse=True)

    i = 0
    j = 0
    baseWords = []

    while(i<n):
        words = TF_IDF_dict[list(scores)[j]]
        for word in words:
            baseWords.append(word)
            i = i+1
            if(i>n):
                break
        j = j+1

        return sentence("base", baseWords, baseWords)

def bestSentence(senteces, base, IDF):
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




answer = "Both versions convey a topic; it’s pretty easy to predict that the paragraph will be about epidemiological evidence, but only the second version establishes an argumentative point and puts it in context. The paragraph doesn’t just describe the epidemiological evidence; it shows how epidemiology is telling the same story as etiology. Similarly, while Version A doesn’t relate to anything in particular, Version B immediately suggests that the prior paragraph addresses the biological pathway (i.e. etiology) of a disease and that the new paragraph will bolster the emerging hypothesis with a different kind of evidence. As a reader, it’s easy to keep track of how the paragraph about cells and chemicals and such relates to the paragraph about populations in different places. A last thing to note about key sentences is that academic readers expect them to be at the beginning of the paragraph. (The first sentence this paragraph is a good example of this in action!) This placement helps readers comprehend your argument. To see how, try this: find an academic piece (such as a textbook or scholarly article) that strikes you as well written and go through part of it reading just the first sentence of each paragraph. You should be able to easily follow the sequence of logic. When you’re writing for professors, it is especially effective to put your key sentences first because they usually convey your own original thinking. It’s a very good sign when your paragraphs are typically composed of a telling key sentence followed by evidence and explanation."

answers = [answer]

links = ["https://www.youtube.com/watch?v=3vku3RvlAdc"]

sentences = []

for answer in answers:
    sentences = sentences + processAns("0", answer)
    
IDF = IDFs(sentences)
TF_IDF_dict = TF_IDF(sentences)

base = buildBase(sentences, TF_IDF_dict, 10)

bestsent = bestSentence(sentences, base, IDF)

summary = makeSummary(sentences, bestsent, base, 100, 0.5, IDF)

final_summary = ""
for sent in summary:
    final_summary = final_summary + "<a href"+ "=" + links[int(sent.id)] + ">" + sent.getOriginalWords() + " </a>"
final_summary = final_summary[:-1]

print('-------------------------------------------------------------ORIGINAL-----------------------------------------------------------------------\n')
print(answer)
print('\n-------------------------------------------------------------SUMMARY----------------------------------------------------------------------\n')
display(HTML(final_summary))