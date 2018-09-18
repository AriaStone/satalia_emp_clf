import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pickle
from collections import Counter
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import scipy.stats
import seaborn as sns
from nltk.stem.porter import PorterStemmer
#
def create_lexicon(texts):
    lex = []
    for i in range(len(texts)):
        print(str(i) + '/483089')
        a = word_tokenize( texts[i].lower())
        lex += a
#    lex1 = []
#    for i in range(len(lex)):
#        if lex[i][0] == "'":
#            lex1.append(lex[i][1:])
#        else:
#            lex1.append(lex[i][:])
    #
    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]
    
    word_count = Counter(lex)
    lex1= []
    for word in word_count:
            if word_count[word] < 4000 and word_count[word] > 5:
                lex1.append(word)
    return lex1

def string_to_vector(lex, text):
    words = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    features = np.zeros(len(lex))
    for word in words:
        if word in lex:
            features[lex.index(word)] += 1
    return features    
###################################################################################################
def transform_weblinks_categories(z): 
# change satalia into satalia, docs google into googledoc, github into github, others into web
    z1 = []
    i = 1    
    for text in z:
        print(str(i) + '/443902')
        i += 1    
        text = re.sub('<https?://.*satalia.*>','sataliahyperlink',text)
        text = re.sub('<https?://.*github.*>','githubhyperlink',text) 
        text = re.sub('<https?://.*google.*>','googledochyperlink',text)    
 
        text = re.sub('<https?.*>','webhyperlink',text) 
        z1.append(text)
    return z1


def transfer_msg_into_wordid_list(z,word_count_index, var_name):
#   var:  ["train_private","train_public","test_private","test_public"]

    texts = transform_weblinks_categories(z)
    len_texts = len(texts)
    for i in range(len_texts):
        print(str(i) + '/ ' +str(len_texts) + 'tokenize texts ' + var_name)
        texts[i] = word_tokenize(texts[i].lower())
    lemmatizer = WordNetLemmatizer()
    for text in texts:
        text = [lemmatizer.lemmatize(word) for word in text]
    lex = list(word_count_index.keys())
    
    for text in range(len_texts):
        print(str(text) + '/' + str(len_texts) + 'transfer into wordid '+ var_name)
    
        for word in range(len(texts[text])):
            if texts[text][word] in lex:
                word_value = texts[text][word]
                texts[text][word] = word_count_index[word_value]
            else:
                texts[text][word] = 1
    return texts

def apply_msg_into_worid_for_all_4sets(train_private, train_public, test_private, test_public):
    test_public_wordid = test_public.copy()
    test_public_wordid.message = transfer_msg_into_wordid_list(test_public.message, word_count_index,"test_public")
    test_private_wordid = test_private.copy()
    test_private_wordid.message = transfer_msg_into_wordid_list(test_private.message,word_count_index, "test_private")
    
    train_public_wordid = train_public.copy()
    train_public_wordid.message = transfer_msg_into_wordid_list(train_public.message,word_count_index, "train_public")
    train_private_wordid = train_private.copy()
    train_private_wordid.message = transfer_msg_into_wordid_list(train_private.message, word_count_index,"train_private")
    return train_private_wordid, train_public_wordid, test_private_wordid, test_public_wordid
#train_private_wordid, train_public_wordid, test_private_wordid, test_public_wordid = apply_msg_into_worid_for_all_4sets(train_private, train_public, test_private, test_public)    


def get_lex_dictionary_for_tf_tfidf_NB(train_private, train_public):
    z = list(train_private.message) + list(train_public.message) 
    texts = transform_weblinks_categories(z)
    lex = []
    for i in range(len(texts)):
        print(str(i) + '/' + str(len(texts)))
        a = word_tokenize(texts[i].lower())
        lex += a
    #    
    for i in range(len(lex)):
        print(str(i) + ' / 6909565')
        if len (lex[i]) > 1:
            if lex[i][0] in ["'","*","+","-","_",".","/"]:
                lex[i] = lex[i][1:]
    
    
    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]
    ##
    word_count = dict(Counter(lex))
    words = list(word_count.keys()) # 139558 words 
    for word in words:
        if word_count[word] > 25000 or word_count[word] <= 5:
            del word_count[word]
    return word_count

#word_count = get_lex_dictionary_for_tf_tfidf_NB(train_private, train_public)
def get_jobBased_tf_tfidf_vector_for_basic_classifier(discipline_msg_train_private,discipline_msg_train_public ):
    jobBased_tf_vector = dict()
    disciplines = list(discipline_msg_train_private.keys())
    for discipline in disciplines:
    #    discipline = 'Software Developer'
        texts = discipline_msg_train_public[discipline] + discipline_msg_train_private[discipline] 
            
        words = list(word_count.keys())
        
        texts = transform_weblinks_categories(texts)   
        topic_lex = []    
        for i in range(len(texts)):
            print(str(i) + '/' + str(len(texts)) + " " + str(discipline) + " step1 tokenization")
            a = word_tokenize(texts[i].lower())
            topic_lex += a    
        for i in range(len(topic_lex)):
            print(str(i) + '/' + str(len(topic_lex)) + " " + str(discipline) + " step2 remove characters")
            if len (topic_lex[i]) > 1:
                if topic_lex[i][0] in ["'","*","+","-","_",".","/"]:
                    topic_lex[i] = topic_lex[i][1:]    
        lemmatizer = WordNetLemmatizer()
        topic_lex = [lemmatizer.lemmatize(word) for word in topic_lex]
        
                
        topic_word_count = dict()
        for word in words:
            topic_word_count[word] = 0            
        
        i = 1
        for word in topic_lex:
            print(str(i) + "/" + str(len(topic_lex)) + " " + str(discipline) + " step3 get term frequency vector")
            i += 1
            if word in words:
                topic_word_count[word]  = topic_word_count[word] + 1
        
        jobBased_tf_vector[discipline] = topic_word_count
        
    jobBased_tfidf_vector = dict()    
    disciplines = list(discipline_msg_train_private.keys())
    for discipline in disciplines:
    #discipline = 'Software Developer'
        jobBased_tfidf_vector[discipline] = dict()
        for word in jobBased_tf_vector[discipline]:
            jobBased_tfidf_vector[discipline][word] = jobBased_tf_vector[discipline][word]/word_count[word]
    return jobBased_tf_vector,jobBased_tfidf_vector

def get_dictionary_index_for_words(word_count):
    word_count_index = dict()
    i = 2
    for word in list(word_count.keys()):
        word_count_index[word] = i
        i += 1
    return word_count_index#test_public_wordid = test_public.copy()
#test_public_wordid.message = transfer_msg_into_wordid_list(test_public.message, word_count_index,"test_public")

#test_private_wordid = test_private.copy()
#test_private_wordid.message = transfer_msg_into_wordid_list(test_private.message,word_count_index, "test_private")
#
#train_public_wordid = train_public.copy()
#train_public_wordid.message = transfer_msg_into_wordid_list(train_public.message,word_count_index, "train_public")
#
#train_private_wordid = train_private.copy()
#train_private_wordid.message = transfer_msg_into_wordid_list(train_private.message, word_count_index,"train_private")
    



#userBased_tf_vector_train, userBased_tfidf_vector_train = get_jobBased_tf_tfidf_vector_for_basic_classifier(discipline_msg_train_private,discipline_msg_train_public)
#userBased_tf_vector_test, userBased_tfidf_vector_test = get_jobBased_tf_tfidf_vector_for_basic_classifier(discipline_msg_test_private,discipline_msg_test_public )


