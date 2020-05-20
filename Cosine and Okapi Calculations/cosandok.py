import nltk
import operator
import math
from nltk.stem import PorterStemmer
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import textmining
import numpy as np
import pandas as pd
import scipy

k1 = 1.2
k2 = k1
b = 0.75
R = 0.0

class OKAPI:
    def __init__(self, corpus):
        idx1 = dict()
        dlt1 = dict()
        for docid in corpus:
            for word in corpus[docid]:
                if str(word) in idx1:
                    if str(docid) in idx1[str(word)]:
                        idx1[str(word)][str(docid)] += 1
                    else:
                        idx1[str(word)][str(docid)] = 1
                else:
                    d = dict()
                    d[str(docid)] = 1
                    idx1[str(word)] = d
            length = len(corpus[docid])
            dlt1[docid] = length
        self.index, self.dlt = idx1, dlt1
        self.postingList()
        
    def run(self, query):
        results = []
        w = []
        dt = []
        qt = []
        res, wi, dti, qti = self.run_query_okapi(query)
        results.append(res)
        w.append(wi)
        dt.append(dti)
        qt.append(qti)
        return results, w, dt, qt

    def run_query_okapi(self, query):
        query_result = dict()
        wi = dict()
        dti = dict()
        qti = dict()
        index = self.index
        for term in query:
            if term in index:
                # retrieve index entry
                doc_dict = index[term]
                # for each document and its word frequency
                for docid, freq in doc_dict.items():
                    # calculate score
                    abc = 0
                    if int(docid) in self.dlt:
                        abc = self.dlt[int(docid)]
                    summ = 0
                    for length in self.dlt.values():
                        summ+=length
                    toty = 0
                    toty = float(summ)/ float(len(self.dlt))
                    score, w, dt, qt = scoreT(n=len(doc_dict), f=freq, qf=1, r=0, N=len(self.dlt), dl=abc, avd_dl=toty)
                    # this document has already been scored once
                    if docid in query_result:
                        query_result[docid] += score
                        wi[docid] += w
                        dti[docid] += dt
                        qti[docid] += qt
                    else:
                        query_result[docid] = score
                        wi[docid] = w
                        dti[docid] = dt
                        qti[docid] = qt
        return query_result, wi, dti, qti

    def postingList(self):
        f = open("postings.txt", "w", encoding='UTF8')
        index = self.index
        for item in index:
            f.write(item + "\n")
            for docID in index[item]:
                tfw = math.log(index[item][docID], 10) + 1
                f.write(docID + "\t" + str(index[item][docID]) + "\t" + str(tfw) + "\n")
        f.close()
        
def scoreT(n, f, qf, r, N, dl, avd_dl):
    K = compute_K(dl, avd_dl)
    first = log(((r + 0.5) / (R - r + 0.5)) / ((n - r + 0.5) / (N - n - R + r + 0.5)))
    second = ((k1 + 1) * f) / (K + f)
    third = ((k2 + 1) * qf) / (k2 + qf)
    return first * second * third, first, second, third

def compute_K(dl, avg_dl):
    return k1 * ((1 - b) + b * (float(dl) / float(avg_dl)))

def parse(filename):
    ps = PorterStemmer()
    tokens = {}
    docToken = {}
    documents = open(filename, encoding='utf8', errors='ignore').read()
    tokens=nltk.word_tokenize(documents)
    tokens = [element.lower() for element in tokens]
    tokens = [ps.stem(element) for element in tokens]
    tokens = [x for x in tokens if x.isalnum()]
    tokens = sorted(tokens)
    singleDoc = documents.split('\n')
    docToken = {}
    for x in range(len(singleDoc)):
        docToken[x] = nltk.word_tokenize(singleDoc[x])
        docToken[x] = [y for y in docToken[x] if y.isalnum()]
        docToken[x] = sorted(docToken[x])
        docToken[x] = [element.lower() for element in docToken[x]]
    return tokens, docToken, documents

def main():
    tdm = textmining.TermDocumentMatrix()
    tokens, docTokens, documents = parse('200_content.txt')
    index = {}
    okapi_sim = OKAPI(corpus=docTokens)

    f = open("dictionary.txt", "a", encoding='UTF8')
    file1 = open('200_content.txt', 'r', encoding='utf8', errors='ignore')
    Lines = file1.readlines()
    for line in Lines:
        f1 = line.strip()
        for p in "!.,:@#$%^&?<>*()[}{]-=;/\"\\\t\n":
             if p in '\n;?:!.,.-':
                 f1 = f1.replace(p, ' ')
             else:
                 f1 = f1.replace(p, '')
        f1 = f1.lower()
        word_tokens = word_tokenize(f1)
        sent = ' '
        for w in word_tokens:
            sent = sent + ' ' + w
        tdm.add_doc(sent)
    count = 0
    mat = [0] * 200
    filename = 'mat.csv'
    tdm.write_csv(filename, cutoff=1)  # writing matrix to excel file
    
    for row in tdm.rows(cutoff=1):  # taking documentword as 2D matrix
        if count == 0:
            print('')
        else:
            mat[count - 1] = row
        count = count + 1

    a = np.array(mat)  # taking list as numpy array
    a = a.transpose()  # converting documentword matrix to word_document matrix
    nonzeros = []  # number of documents the word has appeared in (DF)
    for elem in a:
        nonzeros.append(np.count_nonzero(elem))

    nonzeros = np.divide(200, nonzeros)  # dividing df by 50 (IDF)
    nonzeros = np.log2(nonzeros)  # taking log of IDF
    
    idf = []
    count = 0
    for elem in a:
        idf.append(np.multiply(nonzeros[count], elem))  # taking TF-IDF
        count = count + 1

    idf = np.array(idf)  # taking it as np array
    tfidf = idf.transpose()  # taking transpose to make it documentword matrix
    for i in range(len(tokens) - 1):
        if (tokens[i] != tokens[i+1]):
            f.write(tokens[i] + "\n")
    f.close()

    queryCount = 1
    while True:
        query = input('Enter your query: \n')
        tempQuery = query
        nameOffile1 = "okaquery" + str(queryCount) + "result.txt"
        nameOffile2 = "cosquery" + str(queryCount) + "result.txt"
        nameOffile3 = "cosrelresults.txt"
        nameOffile4 = "okarelresults.txt"
        f = open(nameOffile1,"w+")
        f2 = open(nameOffile2, "w+")
        f3 = open(nameOffile3,"a+")
        f4 = open(nameOffile4,"a+")
        f3.write(tempQuery + "\t")
        f4.write(tempQuery + "\t")

        
        word_tokens = word_tokenize(query)  # breaking query into words
        filtered_sentence = ' '
        for w in word_tokens:
            filtered_sentence = filtered_sentence + ' ' + w  # removing stopwords
        query1 = query
        tdm.add_doc(filtered_sentence)
        tdm.write_csv(filename, cutoff=1)
        data = pd.read_csv("mat.csv")
        query1 = data.iloc[-1].tolist()  # finding vector for query
        query1 = np.multiply(query1, nonzeros)  # getting TF-IDF for query
        queryWeight = 0
        for i in query1:
            if i > 0:
                queryWeight = i

        cosine_similarity = {}
        doc_sum = {}
        doc_num = 1
        
        for elem in tfidf:
            score = 1 - scipy.spatial.distance.cosine(elem, query1)  # function for cosine similarity
            cosine_similarity[doc_num] = score  # key = doc_num, value = score
            doc_sum[doc_num] = sum(elem)
            doc_num = doc_num + 1  #

        result = sorted(cosine_similarity.items(), key=lambda kv: kv[1], reverse=True)
        ij = 0
        for eleme in result:
            if float(eleme[1]) > 0.0:  # comparing the value of cosine scores with alpha for printing
                print(eleme[0], queryWeight, doc_sum[eleme[0]], len(docTokens[eleme[0]]))
                f2.write(tempQuery + "\t" + str(queryWeight) + "\t" + str(doc_sum[eleme[0]]) + "\t" + str(1 / len(docTokens[eleme[0]])) + "\n")
                f2.write(str(eleme[0]) + "\t" + str(eleme[1]) + "\n")
                f3.write(str(eleme[0]) + "\t")
            if (ij == 9):
                break
            ij += 1
        f3.write("\n")
        f2.close()
        
    
        results, w, dt, qt = okapi_sim.run(query)
        for result in results:
            sorted_x = sorted(result.items(), key=operator.itemgetter(1))
            sorted_x.reverse()
            for i in sorted_x[:10]:
                print(i[0], i[1], w[0][str(i[0])], dt[0][str(i[0])], qt[0][str(i[0])])
                f.write(query + "\t" + str(w[0][str(i[0])]) + "\t" + str(dt[0][str(i[0])]) + "\t" + str(qt[0][str(i[0])]) + "\n")
                f.write(str(i[0]) + "\t" + str(i[1]) + "\n")
                f4.write(str(i[0]) + "\t")
            f.close()
        f4.write("\n")

        f3.close()
        f4.close()
        yesno = input('Do you want to continue? press y for yes, n for no\n')
        if yesno == 'n':
            break
        queryCount += 1

main()
