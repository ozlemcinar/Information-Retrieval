# -*- coding: utf-8 -*-

import string
import itertools
from nltk.stem import PorterStemmer


def main():
    ps = PorterStemmer()
    terms = []
    document_id = 0
    terms1 = []
    terms2 = []
    terms3 = []
    terms4 = []
    terms5 = []
    df = 0
    tf = 0
    f = open("posting.txt", "w+")
    f1 = open("dictionary.txt", "w+")
    for line in open('200_title.txt', "r", encoding="utf-8-sig",
                     errors="ignored"):  # opened in text-mode; all EOLs are converted to '\n'
        line = line.lower().split()
        document_id = document_id + 1
        line = [''.join(c for c in s if c not in string.punctuation and c != '‘' and c != '’') for s in
                line]  # noktalamaları atıyor

        for i in line:
            terms.append([ps.stem(i), str(document_id)])

    terms.sort(key=lambda terms: terms[0])

    for row in terms[::-1]:
        if row[0] == '':
            terms.remove(row)
    for i in terms:
        j = i
        for j in terms:
            if i[0] == j[0]:
                df = df + 1
        terms1.append([i, str(df)])
        i = j
        df = 0

    for m in terms1:
        k = m
        for k in terms1:
            if m[0][0] == k[0][0] and m[0][1] == k[0][1]:
                tf = tf + 1

        terms2.append([m, [m[0][1], str(tf)]])
        m = k
        tf = 0
    for i in terms2:
        terms3.append([i[0][0][0], i[0][1], [i[0][0][1], i[1][1]]])  # siralama-->term,df,docid,tf

    for i in terms3:
        for j in terms3:
            if i[0] == j[0]:
                if not [i[0], i[1]] in terms4:
                    terms4.append([i[0], i[1]])

    line = list(terms3 for terms3, _ in itertools.groupby(terms3))  # iki tane 2 varsa bitane indirdi eklemek icin

    numposting = 0
    for i in line:
        a = line[numposting]
        f.write(str(a[2]) + "\n")
        terms5.append([a[0], (a[2])])
        numposting += 1

    num4 = 0
    num5 = 1
    for i in line:
        satir5 = terms4[num5]
        if i[0] == satir5[0]:
            terms4[num5].append(i[2])
            num5 += 1

    for i in line:
        satir4 = terms4[num4]
        x = terms4[num4]
        if i[0] == satir4[0]:
            if i[2] != x[-2:]:
                terms4[num4].append(i[2])
        else:
            num4 += 1


    numdictionary = 0
    counter = 0
    for i in terms4:
        a = terms4[numdictionary]

        if numdictionary != terms.__len__():
            b = terms4[numdictionary - 1]
        if numdictionary == 0:
            f1.write(str(a[0]) + " " + str(a[1]) + " " + str(counter) + "\n")  # bi oncekinin a1yle toplancak
            numdictionary += 1
        else:
            counter = counter + int(b[1])
            f1.write(str(a[0]) + " " + str(a[1]) + " " + str(counter) + "\n")
            numdictionary += 1

    f.close()
    f1.close()


if __name__ == "__main__":
    main()