# -- coding: utf-8 --
import string
import itertools
from nltk.stem import PorterStemmer
import math
import collections

def main():
    ps = PorterStemmer()
    terms=[]
    tfWeight=[]
    lengthNorm = []
    hashTable = [None]*511
    flag = 0
    inpt=[]
    uniqq=[]
    inpt1 = [] #stemlenmis ve ayrilmis liste halinde tutuluyo
    arr = []
    postingValues = []
    sayilar=[]
    idf=[]
    unique_list = []
    overallWeight = []
    duplicated = []
    differentid = []
    differentid1 = []
    LengthOfSentence = []
    wordDfTf = []
    xa = []
    for line in open('dictionary.txt', "r"):  # opened in text-mode; all EOLs are converted to '\n'
        line = line.split()
        terms.append(line)
    for line in open('postings.txt', "r"):  # opened in text-mode; all EOLs are converted to '\n'
        line = line.split()
        postingValues.append(line)
        uniqq.append(line[0])

    for x in uniqq:
        if x not in unique_list:
            unique_list.append(x)
    for i in terms: #dictionary idf
        idf.append([i[0],math.log(unique_list.__len__()/int(i[1]))])
    for line in postingValues:  #  posting weight
        x = math.log(int(line[1]),10) + 1
        tfWeight.append([line[0],x])
    sum = 0
    for count in range(200):
        for line in postingValues:
            if int(line[0]) == count:
                sum = sum + (int(line[1])*int(line[1]))


        lengthNorm.append([count, math.sqrt(sum)])
        sum = 0

    def hashing_func(key):
        return key % len(hashTable)

    def insert(hashTable, key, value):
        hash_key = hashing_func(key)
        hashTable[hash_key] = value

    for line in terms:  # opened in text-mode; all EOLs are converted to '\n'
        randvalue = int((ord(line[0][0]))%58)
        if(hashTable[randvalue]== None):
            insert(hashTable, randvalue, line)
        else:
            while(flag == 0):
                randvalue+=1
                if(randvalue > hashTable.__len__()):
                    randvalue = 0
                if (hashTable[randvalue] == None):
                    insert(hashTable, randvalue, line)
                    flag += 1
            flag = 0

    query=["europe", "stock rally", "debt crisis", "stock future higher"]
    queryNum = 1

    for lik in query:
        nameOfFile = "query" + str(queryNum) + "result" + ".txt"
        file = open(nameOfFile, "w+")
        inpt.append(lik)
        inpt = [''.join(c for c in s if c not in string.punctuation and c != '‘' and c != '’') for s in
                inpt]  # noktalamaları atıyor
        for i in inpt:
            i = i.lower().split()
        for j in i:
            inpt1.append(ps.stem(j))

        for i in inpt1:
            randvalue1 = int((ord(i[0][0])) % 58)
            if (hashTable[randvalue1][0] == i):
                arr.append(hashTable[randvalue1])
            else:
                while (flag == 0):
                    randvalue1 += 1
                    if (randvalue1 > hashTable.__len__()):
                        randvalue1 = 0
                    if (hashTable[randvalue1][0] == i):
                        arr.append(hashTable[randvalue1])
                        flag += 1
                flag = 0
        #butun posting values degerleri tutuluyo, ikili halde
        flagg = 0
        for i in arr:
            x = 0
            while x < int(i[1]):
                while flagg == 0:
                    for s in idf:
                        if(s[0] == i[0]):
                            kelime = s[1]
                            flagg = 1
                sayilar.append([i[0],i[1],tfWeight[int(i[2])+x],kelime]) ## tf, idf sirayla
                flagg = 0
                x+=1
        counter = 0
        counter2 = 0
        sum2 =0

        while counter < sayilar.__len__():
            while counter2 < (int(sayilar[counter][1])+counter):
                sum2 = sum2 + (sayilar[counter2][2][1]*sayilar[counter2][3])
                counter2+=1
            counter=counter2
            overallWeight.append(sum2)
            sum2 = 0


        #for i in sayilar:
         #   print(i)

        for i in sayilar:
            duplicated.append(i[2][0])
        duplicated.sort()
        differentid.append([item for item, count in collections.Counter(duplicated).items() if count >= 1])
        for i in differentid:
            differentid1.append(i)

        #for i in differentid:
         #   print(i)
        say = 0
        tf = 0
        while say<differentid[0].__len__():
            lengthOfSentence = 0
            lineCount = -1
            xa = []
            for i in postingValues:
                lineCount += 1
                tf = i[1]
                if i[0] ==  differentid[0][say] :
                    lengthOfSentence+=1
                    x = lineCount
                    xa.append([x,tf,i[0]])
            LengthOfSentence.append([differentid[0][say],lengthOfSentence,xa,xa[2]])
            #print(i[0])
            say+=1

        for i in LengthOfSentence:
            say = 0
            while say< (i[2].__len__()):
                for k in terms:
                    if int(k[2])<= i[2][say][0]:
                        min = k[0]
                        mindf = k[1]
                        mintf = i[2][say][1]
                        minID = i[3]
                wordDfTf.append([say,min,mindf,mintf,minID])

                say+=1
        i = 0
        sum = 0
        index=0
        sifir_array=[]
        for i in wordDfTf:

            if i[0] == 0:
                sifir_array.append(index)
            index = index + 1
        count=0
        z = 1
        sum = 0
        deneme2 = []
        sayy = 0
        while z < int(sifir_array.__len__()):
            while count< (sifir_array[z]):
                deneme2.append([z-1,math.log(200/int(wordDfTf[count][2]),10) * (1+math.log(int(wordDfTf[count][3]),10)),wordDfTf[count][4][2]])
                count = count + 1
            if z != int(sifir_array.__len__()):
                sum = 0
                z+=1

        if z == int(sifir_array.__len__()):
            while count < wordDfTf.__len__():
                deneme2.append([z-1,math.log(200/int(wordDfTf[count][2]),10) * (1+math.log(int(wordDfTf[count][3]),10)),wordDfTf[count][4][2]])
                count = count + 1


        if(deneme2.__len__() > overallWeight.__len__()):
            lenn = overallWeight.__len__()
        else:
            lenn = deneme2.__len__()
        counts = 0
        o = 0
        total = 0.0
        sonliste = []
        for i in deneme2:
            c = 0
            if i[0] == o:
                while c<overallWeight.__len__():
                    total = total +float(overallWeight[c]) * float(deneme2[o][1])
                    c+=1
                sonliste.append([total,deneme2[o][2]])
            else:
                o+=1
            total = 0.0
        sonliste.sort()


        unique_list = []

        for x in sonliste:
            if x not in unique_list:
                unique_list.append(x)

        unique_list.reverse()
        count55 = 0
        for i in unique_list:
            if unique_list.__len__() >= 10 and count55 < 10:
                count55 = count55 + 1
                file.write(str(i[0]) + "\t" + str(i[1])+"\n")
            if unique_list.__len__() < 10:
                file.write(str(i[0]) + "\t" + str(i[1])+"\n")
        queryNum+=1

if __name__ == "__main__":
    main()