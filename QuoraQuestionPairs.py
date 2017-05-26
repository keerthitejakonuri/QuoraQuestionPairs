# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 16:59:36 2017

@author: keert
"""

import math
import pandas as pd
import re
from nltk.corpus import wordnet
from nltk.corpus import brown
from nltk import FreqDist

freq = FreqDist(brown.words())
log_words = math.log(len(brown.words()) + 1)

#This function is used to calculate the similarity between two words
#It takes a word, sentence vector and value as input parameters
#It returna a list indicating similarity and index of word matched
def similarity(term, sc1, factor):
    try:
        if sc1 == [] or sc1 == None:
            return [0, None]
        sim_list = []
        for a in sc1:
            v1 = wordnet.synsets(term)
            v2 = wordnet.synsets(a)
            if v1 == [] or v2 == []:
                return [0, None]
            else:
                v1 = v1[0]
                v2 = v2[0]
            if v1.path_similarity(v2) != None:
                sim_list.append(v1.path_similarity(v2) * v1.wup_similarity(v2))
            else:
                sim_list.append(0)
        if max(sim_list) >= factor:
            return [max(sim_list), sim_list.index(max(sim_list))]
        else:
            return [0, None]
    except:
        print(sc1)

#This method is used to compute semantic vector of a given sentence
#It takes sentence vector and jointset as input parameters
#It returns a semantic vector of a sentence

def similarityVector(sc1, joinset):
    vector = []
    identity = 0
    for term in joinset:
        identity = 1 - (math.log(freq[term] + 1)/log_words)
        if term in sc1:
            vector.append(1 * identity * identity)
        else:
            sim, termPos = similarity(term,sc1, 0.2)
            if termPos == None:
                identity = 0
            else:
                identity = identity * (1 - (math.log(freq[sc1[termPos]] + 1)/log_words))
            vector.append(sim * identity)
    return vector

#This method measures the cosine similarity between two vectors
#This method takes semantic vector of two sentences as input parameters
#It returns a value between 0 and 1
def cosineSimilarity(v1,v2):
    mod_v1 = 0
    mod_v2 = 0
    dot_product = 0
    for i in range(len(v1)):
        mod_v1 = mod_v1 + v1[i]*v1[i]
        mod_v2 = mod_v2 + v2[i]*v2[i]
        dot_product = dot_product + v1[i]*v2[i]
        if mod_v1 == 0 or mod_v2 == 0:
            return 0    
    return (dot_product/math.sqrt(mod_v1*mod_v2))

#This method measures the Normalized Euclidean distance similarity between two vectors
#This method takes semantic vector of two sentences as input parameters
#It returns a value between 0 and 1
def euclideanDistance(v1,v2):
    numerator = 0
    denominatorX = 0
    denominatorY = 0
    for i in range(len(v1)):
        numerator += (v1[i]-v2[i])*(v1[i]-v2[i])
        denominatorX += (v1[i]*v1[i])
        denominatorY += (v2[i]*v2[i])
    return (math.sqrt(numerator)/(math.sqrt(denominatorX)*math.sqrt(denominatorY)))
#This method measures the Normalized Manhattan distance similarity between two vectors
#This method takes semantic vector of two sentences as input parameters
#It returns a value between 0 and 1
def manhattanDistance(v1,v2):
    numerator = 0
    denominatorX = 0
    denominatorY = 0
    for i in range(len(v1)):
        numerator += abs((v1[i]-v2[i]))
        denominatorX += (v1[i]*v1[i])
        denominatorY += (v2[i]*v2[i])
    return (math.sqrt(numerator)/(math.sqrt(denominatorX)*math.sqrt(denominatorY)))
    
#This function determines the Word order vector of a sentence
#It takes sentence vector and joint set as input parameters    
#It returns an order vector for sentence
def orderVector(sc1, joinset):
    vector = []
    indexPos = None
    for term in joinset:
        if term in sc1:
            vector.append(sc1.index(term))
        else:
            indexPos = similarity(term, sc1, 0.4)[1]
            if indexPos == None:
                vector.append(0)
            else:
                vector.append(indexPos)
    return vector

#This method measures order similarity between two vectors
#This method takes order vector of two sentences as input parameters
#It returns a value between 0 and 1
def orderSimilarity(v1, v2):
    mod1 = 0
    mod2 = 1
    for i in range(len(v1)):
        mod1 = mod1 + (v1[i] - v2[i]) * (v1[i] - v2[i])
        mod2 = mod2 + (v1[i] + v2[i]) * (v1[i] + v2[i])
    return 1 - math.sqrt(mod1/mod2)
#data preprocessing   
dataset = pd.read_csv('train.csv')
dataset['question1'] = dataset['question1'].replace("?", "")
dataset['question2'] = dataset['question2'].replace("?", "")
dataset = dataset[pd.notnull(dataset['question1'])]
dataset = dataset[pd.notnull(dataset['question2'])]
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 5].values

#removing unnecessary characters
for i in range(len(X)):
    X[i][3] = X[i][3].replace('?',"")
    X[i][3] = re.sub('[?*<>]','',X[i][3])
    X[i][3] = X[i][3].replace("/"," ").replace("'s"," ").replace("'","").replace(","," ").replace("-"," ").lower()
    X[i][3] = re.sub('[().]','',X[i][3])
    X[i][3] = X[i][3].strip()
    X[i][4] = X[i][4].replace('?',"")
    X[i][4] = re.sub('[?*<>]','',X[i][4])
    X[i][4] = X[i][4].replace("/"," ").replace("'s"," ").replace("'","").replace(","," ").replace("-"," ").lower()
    X[i][4] = re.sub('[().]','',X[i][4])
    X[i][4] = X[i][4].strip()
result = 0
count = 0
total = 0
for k in range(0, 10000):
    total = total + 1
    print(total)
    a1 = X[k][3].split()
    a2 = X[k][4].split()

    joinset = set()
    for i in range(len(a1)):
        a1[i] = a1[i].lower()
        joinset.add(a1[i])
    for i in range(len(a2)):
        a2[i] = a2[i].lower()
        joinset.add(a2[i])
    joinset = list(joinset)
    sem_vector1 = []
    sem_vector2 = []
    order_vector1 = []
    order_vector2 = []
    try:
        sem_vector1 = similarityVector(a1, joinset)
        sem_vector2 = similarityVector(a2, joinset)
        
        order_vector1 = orderVector(a1, joinset)
        order_vector2 = orderVector(a2, joinset)
    except:
        print(a1)
        print(a2)
        print(k)
    
    #print(joinset)
    #print(sem_vector1)
    #print(sem_vector2)
    #print(order_vector1)
    #print(order_vector2)
    semantic = cosineSimilarity(sem_vector1, sem_vector2)
    order = orderSimilarity(order_vector1, order_vector2)
    #print(semantic)
    #print(order)
    delta = 0.85
    similarity_final = delta * semantic + order * (1 - delta)   
    #print(similarity_final)
    
    if similarity_final >= 0.75:
        result = 1
    else:
        result = 0
    if result == Y[k]:
        count = count + 1

print("The Accuracy of this model is:",count/10000)













'''


Performance Evaluation: The various Performance Evaluation Metrics used are:
    
    
    
Accuracy = (Total number of correct predictions)/(Total number of records)



Precision = (True Positive)/(True Positive + False Positive), from the Confusion Matrix

            

Recall = (True Positive)/(True Positive + False Negative), from the Confusion Matrix

         

F-measure = (2*True Positive)/((2*True Positive) + False Positive + False Negative)


'''
