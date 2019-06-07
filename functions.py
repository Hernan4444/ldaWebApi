import lda
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from string import punctuation

nltk.download('punkt')
non_letters = list(punctuation)

#we add spanish punctuation
non_letters.extend(['¿', '¡', '-'])
non_letters.extend(map(str,range(10)))


# based on http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
stemmer = SnowballStemmer('spanish')
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def load_file(filename, json_data):
    for files in json_data["files"]:
        if files[0] == filename:
            with open(os.path.join("data", files[2]), encoding="UTF-8") as file:
                return file.readlines()

def load_database(option=None):

    if option == 5:
        categ = ['alt.atheism', 'comp.graphics', 'rec.autos', 'rec.sport.hockey', 'sci.med']
    elif option == 10:
        categ = ['alt.atheism', 'comp.graphics', 'comp.windows.x', 'misc.forsale', 'rec.autos',
                 'rec.sport.hockey', 'sci.crypt', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', ]
    else:
        categ = None


    return fetch_20newsgroups(data_home="data/", subset='train', remove=(
        'headers', 'footers', 'quotes'), categories=categ).data


def run_lda(dataset, iterations, alpha, eta, topics, stop_words, stop_words_arrays, stemming):

    new_dataset = []
    if len(stop_words_arrays):
        for line in dataset:
            # remove non letters
            text = ''.join([c.lower() for c in line if c.lower() not in non_letters])
            text = ' '.join([c.lower() for c in text.split(" ") if c.lower() not in stop_words_arrays])
            # tokenize
            tokens =  word_tokenize(text)
            line = " ".join(tokens)
            if stemming:
                try:
                    stems = stem_tokens(tokens, stemmer)
                except Exception:
                    stems = ['']
                line = " ".join(stems)

            new_dataset.append(line)
    else:
        new_dataset = dataset

    if stop_words:
        tf_vectorizer = CountVectorizer(stop_words='english', max_features=1000, min_df=2, max_df=0.9)
    else:
        tf_vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.9)

    tf = tf_vectorizer.fit_transform(new_dataset)

    model = lda.LDA(n_topics=topics, n_iter=iterations, alpha=alpha, eta=eta, refresh=50)
    model.fit(tf)

    word_topics = []
    for topic_dist in model.topic_word_:
        word_topics.append(list(zip(
            ["{:0.5f}".format(x) for x in np.array(topic_dist)[np.argsort(topic_dist)][:-20:-1]],
            np.array(tf_vectorizer.get_feature_names())[np.argsort(topic_dist)][:-20:-1]
        )))

    doc_topic = []
    for doc_dist, doc in zip(model.doc_topic_, dataset):
        doc_topic.append(["{:0.5f}".format(x) for x in doc_dist] + [doc.replace("\n", " ")])


    return {"word_topic": word_topics, "doc_topic": doc_topic}


def run_interactive_lda(dataset, iterations, alpha, eta, nu, topics, seeds, mode, stop_words, stop_words_arrays, stemming):

    if stop_words:
        tf_vectorizer = CountVectorizer(stop_words='english', max_features=1000, min_df=2, max_df=0.9)
    else:
        tf_vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.9)

    new_dataset = []
    if len(stop_words_arrays):
        for line in dataset:
            # remove non letters
            text = ''.join([c.lower() for c in line if c.lower() not in non_letters])
            text = ' '.join([c.lower() for c in text.split(" ") if c.lower() not in stop_words_arrays])
            # tokenize
            tokens =  word_tokenize(text)
            line = " ".join(tokens)
            if stemming:
                try:
                    stems = stem_tokens(tokens, stemmer)
                except Exception:
                    stems = ['']
                line = " ".join(stems)

            new_dataset.append(line)
    else:
        new_dataset = dataset

    tf = tf_vectorizer.fit_transform(new_dataset)
    vocab = tf_vectorizer.get_feature_names()

    new_seeds = []
    # print(seeds)

    for constrain in seeds:
        new_topic = []
        for word in seeds[constrain]:
            if word in vocab:
                new_topic.append(vocab.index(word))
            # else:
            #     print(word, constrain)
        new_seeds.append(new_topic)
    # print(new_seeds)

    model = lda.LDA(n_topics=topics, n_iter=iterations, alpha=alpha,
                    eta=eta, nu=nu, seed=new_seeds, refresh=100, mode=mode.lower())
                    
    model.fit(tf)

    word_topics = []
    for topic_dist in model.topic_word_:
        word_topics.append(list(zip(
            ["{:0.5f}".format(x) for x in np.array(topic_dist)[np.argsort(topic_dist)][:-20:-1]],
            np.array(tf_vectorizer.get_feature_names())[np.argsort(topic_dist)][:-20:-1]
        )))

    doc_topic = []
    for doc_dist, doc in zip(model.doc_topic_, dataset):
        doc_topic.append(["{:0.5f}".format(x) for x in doc_dist] + [doc.replace("\n", " ")])

    return {"word_topic": word_topics, "doc_topic": doc_topic}
