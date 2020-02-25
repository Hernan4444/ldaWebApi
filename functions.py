import lda
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import os
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
import json
from string import punctuation, ascii_uppercase
from collections import defaultdict
import time

STEMMER = SnowballStemmer('spanish')
NON_LETTERS = list(punctuation)

# we add spanish punctuation
NON_LETTERS.extend(['¿', '¡', '-'])
NON_LETTERS.extend(map(str, range(10)))


def filter_df(df, words, wordsOr, indexs):
    if len(words) == 0 and len(wordsOr) == 0:
        return df

    indexs_doc = None
    for word in wordsOr:
        stemmer_word = STEMMER.stem(word)

        if stemmer_word != "" and stemmer_word in indexs:
            if indexs_doc is None:
                indexs_doc = set(indexs[stemmer_word])
            else:
                indexs_doc = indexs_doc.union(set(indexs[stemmer_word]))

    for word in words:
        stemmer_word = STEMMER.stem(word)

        if stemmer_word != "" and stemmer_word in indexs:
            if indexs_doc is None:
                indexs_doc = set(indexs[stemmer_word])
            else:
                indexs_doc = indexs_doc.intersection(set(indexs[stemmer_word]))
        else:
            indexs_doc = set()

    if indexs_doc is None:
        indexs_doc = []
    return df.iloc[list(indexs_doc)]


def run_lda(dataset, iterations, alpha, eta, topics, is_encuesta, stopword, stemming, nu=0.004, seeds=[], mode=None):

    if stopword and stemming:
        new_dataset = dataset["TEXTO_STOPWORD_STEMMING"].values
    elif stemming:
        new_dataset = dataset["TEXTO_STEMMING"].values
    elif stopword:
        new_dataset = dataset["TEXTO_STOPWORD"].values
    else:
        new_dataset = dataset["TEXTO"].values

    new_seeds = []

    tf_vectorizer = CountVectorizer(max_features=1000, min_df=0, max_df=0.9)
    tf = tf_vectorizer.fit_transform(new_dataset)
    vocab = tf_vectorizer.get_feature_names()

    for constrain in seeds:
        new_topic = []
        for word in seeds[constrain]:
            if word in vocab:
                new_topic.append(vocab.index(word))

        new_seeds.append(new_topic)

    model = lda.LDA(n_topics=topics, n_iter=iterations, alpha=alpha,
                    eta=eta, refresh=50, seed=new_seeds, mode=mode, nu=nu)
    model.fit(tf)

    word_topics = []
    for topic_dist in model.topic_word_:
        word_topics.append(list(zip(
            ["{:0.3f}".format(x) for x in np.array(topic_dist)[np.argsort(topic_dist)][:-20:-1]],
            np.array(tf_vectorizer.get_feature_names())[np.argsort(topic_dist)][:-20:-1]
        )))

    # TODO diferenciar entre encuesta y otro texto
    doc_topic = []
    teachers_options = set()
    sigles_options = set()
    teachers_filter = defaultdict(lambda: set())
    sigles_filter = defaultdict(lambda: set())
    for index, (doc_dist, (_, doc)) in enumerate(zip(model.doc_topic_, dataset.iterrows())):
        data = {
            'dist': ["{:0.3f}".format(x) for x in doc_dist],
            'text': process_text(doc["TEXTO"]),
            'index': index,
            'metadata': {
                'profesor': doc["NOMBRE_DEL_DOCENTE"],
                'año': doc["ANO_APLICACION"],
                'semestre': doc["SEMESTRE"],
                'sigla': doc["SIGLA"],
                'pregunta': doc["PREGUNTA"]
            }
        }
        teachers_filter[doc["NOMBRE_DEL_DOCENTE"]].add(doc["SIGLA"])
        sigles_filter[doc["SIGLA"]].add(doc["NOMBRE_DEL_DOCENTE"])

        doc_topic.append(data)
        teachers_options.add(doc["NOMBRE_DEL_DOCENTE"])
        sigles_options.add(doc["SIGLA"])

    return {
        "word_topic": word_topics,
        "doc_topic": doc_topic,
        "filter": {
            'teacher': list(teachers_options),
            'sigle': list(sigles_options),
            "teacher_filter": {x: list(teachers_filter[x]) for x in teachers_filter},
            "sigle_filter": {x: list(sigles_filter[x]) for x in sigles_filter}
        }
    }


def process_text(doc):
    data = []
    if isinstance(doc, float):
        print(doc, "##")
    new_doc = str(doc).replace("\n", " ")
    for word in new_doc.split(" "):
        data.append({"text": word, "class": 'black'})
    return data


def bad_and_good_word(index, text, mark_words_index, word_types):
    data = []
    if index not in mark_words_index:
        for word in text.split(" "):
            data.append({"text": word, "class": 'black'})
        return data

    for word in text.split(" "):
        aux_word = ''.join([c.lower() for c in word if c.lower() not in NON_LETTERS])
        class_ = 'black'
        aux_word = STEMMER.stem(aux_word)
        for marked_word in mark_words_index[index]:
            if aux_word.startswith(marked_word):
                class_ = 'red'
                if word_types[marked_word] == 'good':
                    class_ = 'green'
                break
        data.append({"text": word, "class": class_})

    return data


def get_index(good_words, bad_words, indexs, doc_index):
    doc_indexs = defaultdict(set)
    words_type = {}
    for word in good_words:
        if word in indexs:
            for index in indexs[word]:
                if index in doc_index:
                    doc_indexs[index].add(word)
            words_type[word] = "good"

    for word in bad_words:
        if word in indexs:
            for index in indexs[word]:
                if index in doc_index:
                    doc_indexs[index].add(word)
            words_type[word] = "bad"

    return doc_indexs, words_type


def search_by_words(words, wordsOr, df, good_words, bad_words, teacher, sigle, indexs):
    data = []
    documents = filter_df(df, words, wordsOr, indexs)
    good_words = [STEMMER.stem(word).lower() for word in good_words]
    bad_words = [STEMMER.stem(word).lower() for word in bad_words]
    mark_words_index, word_types = get_index(good_words, bad_words, indexs, documents.index)

    teachers_filter = defaultdict(lambda: set())
    sigles_filter = defaultdict(lambda: set())
    
    teachers = set()
    sigles = set()
    for index, doc in documents.iterrows():
        doc = doc.to_dict()

        if teacher != '' and doc['metadata']['profesor'].lower() != teacher.lower():
            continue

        if sigle != '' and doc['metadata']['sigla'].lower() != sigle.lower():
            continue

        doc["text"] = bad_and_good_word(index, doc["text"], mark_words_index, word_types)
        data.append(doc)

        teachers.add(doc['metadata']['profesor'])
        sigles.add(doc['metadata']['sigla'])

        teachers_filter[doc['metadata']['profesor']].add(doc['metadata']['sigla'])
        sigles_filter[doc['metadata']['sigla']].add(doc['metadata']['profesor'])

    return {
        "data": data,
        "teacher_filter": {x: list(teachers_filter[x]) for x in teachers_filter},
        "sigle_filter": {x: list(sigles_filter[x]) for x in sigles_filter},
        "teachers": list(teachers),
        "sigles": list(sigles)

    }
