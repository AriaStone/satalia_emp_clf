from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

def tfidf_vec(doc_set):
    docs = doc_set
    docs1 = []
    for doc in docs:
        docs1.append(doc.split())
    
    dct = Dictionary(docs1)
    corpus = [dct.doc2bow(line) for line in docs1]
    model = TfidfModel(corpus)
    vectors = []
    for i in range(len(docs)):
        vector = model[corpus[i]]
        vectors.append(vector)
    return vectors

def data4_lda_tfidf_model(doc_set):
    
    tokenizer = RegexpTokenizer(r'\w+')
        
    # create English stop words list
    en_stop = get_stop_words('en')
    
    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    
    # list for tokenized documents in loop
    texts = []
    counter = 1
    #loop through document list
    for i in doc_set:
        print(str(counter) + '/' + str(discipline))
        counter += 1
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
    
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        
        # add tokens to list
        texts.append(stemmed_tokens)
    
    texts = [x for x in texts if x != []]
    
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)    
    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]
    #
    model = TfidfModel(corpus)
    corpus_tfidf = []
    for i in range(len(texts)):
        vector = model[corpus[i]]
        corpus_tfidf.append(vector)
    return corpus, corpus_tfidf, texts # texts should use dictionary = corpora.Dictionary(texts) to convert it into dictionary then as input for lda

def get_topicBased_matrix_for_LDA(discipline_msg_train_public,discipline_msg_train_private):
    corpus = dict()
    corpus_tfidf = dict()
    texts = dict()
    #
    for discipline in list(discipline_msg_test_private.keys()):
    
        doc_set = discipline_msg_train_public[discipline] + discipline_msg_train_private[discipline]
        corpus[discipline], corpus_tfidf[discipline], texts[discipline] = data4_lda_tfidf_model(doc_set)
    del discipline_msg_test_private
    del discipline
    del discipline_msg_test_public
    del discipline_msg_train_private
    del discipline_msg_train_public
    
    return corpus, corpus_tfidf, texts

def apply__topicBased_lda(corpus, corpus_tfidf, texts): 
    
    topicBased_LDA_tf = dict()
    topicBased_LDA_tfidf = dict()
    
    for discipline in list(corpus.keys()):
        
        #discipline = 'Executive'
        topic_texts = texts[discipline]
        topic_corpus = corpus[discipline]
        topic_corpus_tfidf = corpus_tfidf[discipline]
        dictionary = corpora.Dictionary(topic_texts)
        
        print("user " + str(discipline) + ' tf')
#        ldamodel = gensim.models.ldamodel.LdaModel(topic_corpus, num_topics=10, id2word = dictionary, passes=20)
#        topic_words = ldamodel.print_topics(10,num_words=50)
        lsi = gensim.models.lsimodel.LsiModel(corpus=topic_corpus, id2word=dictionary, num_topics=50)
        topic_words = lsi.print_topics(10)
        topicBased_LDA_tf[discipline] = topic_words
        
        print("user " + str(discipline) + ' tfidf')
#        ldamodel_tfidf1 = gensim.models.ldamodel.LdaModel(topic_corpus_tfidf, num_topics=10, id2word = dictionary, passes=20)
#        topic_words_tfidf = ldamodel_tfidf.print_topics(10,num_words=50)
        lsi_tfidf = gensim.models.lsimodel.LsiModel(corpus=topic_corpus_tfidf, id2word=dictionary, num_topics=50)
        topic_words_tfidf = lsi.print_topics(10)
        topicBased_LDA_tfidf[discipline] = topic_words_tfidf
    return topicBased_LDA_tf, topicBased_LDA_tfidf

#topicBased_lsi_tf, topicBased_lsi_tfidf = apply__topicBased_lda(corpus, corpus_tfidf, texts)