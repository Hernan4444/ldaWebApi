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
import pandas as pd


nltk.download('punkt')
non_letters = list(punctuation)

# we add spanish punctuation
non_letters.extend(['¿', '¡', '-'])
non_letters.extend(map(str, range(10)))


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
            return pd.read_csv(os.path.join("data", files[2]), encoding="UTF-8", sep="\t").fillna('')


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


def run_lda(dataset, iterations, alpha, eta, topics, english_stopwords, stop_words_arrays, stemming):

    new_dataset = []
    if len(stop_words_arrays):
        for line in dataset["TEXTO"].values:
            # remove non letters
            text = ''.join([c.lower() for c in line if c.lower() not in non_letters])
            text = ' '.join([c.lower() for c in text.split(
                " ") if c.lower() not in stop_words_arrays])
            # tokenize
            tokens = word_tokenize(text)
            line = " ".join(tokens)
            if stemming:
                try:
                    stems = stem_tokens(tokens, stemmer)
                except Exception:
                    stems = ['']
                line = " ".join(stems)

            new_dataset.append(line)
    else:
        new_dataset = dataset["TEXTO"].values

    if english_stopwords:
        tf_vectorizer = CountVectorizer(
            stopwords='english', max_features=1000, min_df=0, max_df=0.9)
    else:
        tf_vectorizer = CountVectorizer(max_features=1000, min_df=0, max_df=0.9)

    tf = tf_vectorizer.fit_transform(new_dataset)

    model = lda.LDA(n_topics=topics, n_iter=iterations, alpha=alpha, eta=eta, refresh=50)
    model.fit(tf)

    word_topics = []
    for topic_dist in model.topic_word_:
        word_topics.append(list(zip(
            ["{:0.3f}".format(x) for x in np.array(topic_dist)[np.argsort(topic_dist)][:-20:-1]],
            np.array(tf_vectorizer.get_feature_names())[np.argsort(topic_dist)][:-20:-1]
        )))

    doc_topic = []
    teachers_options = set()
    sigles_options = set()
    for doc_dist, (_, doc) in zip(model.doc_topic_, dataset.iterrows()):
        data = {
            'dist': ["{:0.3f}".format(x) for x in doc_dist],
            'text': process_text(doc["TEXTO"]),
            'metadata': {
                'profesor': doc["NOMBRE_DEL_DOCENTE"],
                'año': doc["ANO_APLICACION"],
                'semestre': "S/I",
                'sigla': doc["SIGLA"],
                'pregunta': doc["PREGUNTA"],
            }
        }
        doc_topic.append(data)
        teachers_options.add(doc["NOMBRE_DEL_DOCENTE"])
        sigles_options.add(doc["SIGLA"])

    return {"word_topic": word_topics, "doc_topic": doc_topic, "filter": {
        'teacher': list(teachers_options),
        'sigle': list(sigles_options)
    }}


def process_text(doc):
    data = []
    new_doc = doc.replace("\n", " ")
    for word in new_doc.split(" "):
        data.append({"text": word, "class": 'black'})

    return data


def bad_and_good_word(doc, good_words, bad_words):
    data = []
    for word in doc:
        class_ = 'black'
        lower_word = word['text'].lower()
        if lower_word in bad_words:
            class_ = 'red'
        elif lower_word in good_words:
            class_ = 'green'

        data.append({"text": word['text'], "class": class_})

    return data


def run_interactive_lda(dataset, iterations, alpha, eta, nu, topics, seeds, mode, english_stopwords, stop_words_arrays, stemming):

    new_dataset = []
    if len(stop_words_arrays):
        for line in dataset["TEXTO"].values:
            # remove non letters
            text = ''.join([c.lower() for c in line if c.lower() not in non_letters])
            text = ' '.join([c.lower() for c in text.split(
                " ") if c.lower() not in stop_words_arrays])
            # tokenize
            tokens = word_tokenize(text)
            line = " ".join(tokens)
            if stemming:
                try:
                    stems = stem_tokens(tokens, stemmer)
                except Exception:
                    stems = ['']
                line = " ".join(stems)

            new_dataset.append(line)
    else:
        new_dataset = dataset["TEXTO"].values

    if english_stopwords:
        tf_vectorizer = CountVectorizer(
            stopwords='english', max_features=1000, min_df=0, max_df=0.9)
    else:
        tf_vectorizer = CountVectorizer(max_features=1000, min_df=0, max_df=0.9)

    tf = tf_vectorizer.fit_transform(new_dataset)
    new_seeds = []
    vocab = tf_vectorizer.get_feature_names()

    for constrain in seeds:
        new_topic = []
        for word in seeds[constrain]:
            if word in vocab:
                new_topic.append(vocab.index(word))
            # else:
            #     print(word, constrain)
        new_seeds.append(new_topic)

    model = lda.LDA(n_topics=topics, n_iter=iterations, alpha=alpha,
                    eta=eta, nu=nu, seed=new_seeds, refresh=100, mode=mode.lower())

    model.fit(tf)

    word_topics = []
    for topic_dist in model.topic_word_:
        word_topics.append(list(zip(
            ["{:0.3f}".format(x) for x in np.array(topic_dist)[np.argsort(topic_dist)][:-20:-1]],
            np.array(tf_vectorizer.get_feature_names())[np.argsort(topic_dist)][:-20:-1]
        )))

    doc_topic = []
    teachers_options = set()
    sigles_options = set()
    for doc_dist, (_, doc) in zip(model.doc_topic_, dataset.iterrows()):
        data = {
            'dist': ["{:0.3f}".format(x) for x in doc_dist],
            'text': process_text(doc["TEXTO"]),
            'metadata': {
                'profesor': doc["NOMBRE_DEL_DOCENTE"],
                'año': doc["ANO_APLICACION"],
                'semestre': "S/I",
                'sigla': doc["SIGLA"],
                'pregunta': doc["PREGUNTA"],
            }
        }
        doc_topic.append(data)
        teachers_options.add(doc["NOMBRE_DEL_DOCENTE"])
        sigles_options.add(doc["SIGLA"])

    return {"word_topic": word_topics, "doc_topic": doc_topic, "filter": {
        'teacher': list(teachers_options),
        'sigle': list(sigles_options)
    }}


    # if english_stopwords:
    #     tf_vectorizer = CountVectorizer(
    #         english_stopwords='english', max_features=1000, min_df=2, max_df=0.9)
    # else:
    #     tf_vectorizer = CountVectorizer(max_features=1000, min_df=2, max_df=0.9)

    # new_dataset = []
    # if len(stop_words_arrays):
    #     for line in dataset:
    #         # remove non letters
    #         text = ''.join([c.lower() for c in line if c.lower() not in non_letters])
    #         text = ' '.join([c.lower() for c in text.split(
    #             " ") if c.lower() not in stop_words_arrays])
    #         # tokenize
    #         tokens = word_tokenize(text)
    #         line = " ".join(tokens)
    #         if stemming:
    #             try:
    #                 stems = stem_tokens(tokens, stemmer)
    #             except Exception:
    #                 stems = ['']
    #             line = " ".join(stems)

    #         new_dataset.append(line)
    # else:
    #     new_dataset = dataset

    # tf = tf_vectorizer.fit_transform(new_dataset)
    # vocab = tf_vectorizer.get_feature_names()

    # new_seeds = []
    # # print(seeds)

    # for constrain in seeds:
    #     new_topic = []
    #     for word in seeds[constrain]:
    #         if word in vocab:
    #             new_topic.append(vocab.index(word))
    #         # else:
    #         #     print(word, constrain)
    #     new_seeds.append(new_topic)
    # # print(new_seeds)

    # model = lda.LDA(n_topics=topics, n_iter=iterations, alpha=alpha,
    #                 eta=eta, nu=nu, seed=new_seeds, refresh=100, mode=mode.lower())

    # model.fit(tf)

    # word_topics = []
    # for topic_dist in model.topic_word_:
    #     word_topics.append(list(zip(
    #         ["{:0.3f}".format(x) for x in np.array(topic_dist)[np.argsort(topic_dist)][:-20:-1]],
    #         np.array(tf_vectorizer.get_feature_names())[np.argsort(topic_dist)][:-20:-1]
    #     )))

    # doc_topic = []
    # for doc_dist, doc in zip(model.doc_topic_, dataset):
    #     doc_topic.append(["{:0.3f}".format(x) for x in doc_dist] + [doc.replace("\n", " ")])

    # return {"word_topic": word_topics, "doc_topic": doc_topic}


def join_text(array):
    return " ".join([x["text"] for x in array])


def search_by_words(word, topic, documents, good_words, bad_words):
    data = [x for x in documents if word in join_text(x['text'])]
    for i in range(len(data)):
        data[i]["text"] = bad_and_good_word(data[i]["text"], good_words, bad_words)
    return sorted(data, key=lambda x: x['dist'][topic], reverse=True)


def search_by_word_and_topic(word, topic, documents, good_words, bad_words):
    data = [x for x in documents if word in join_text(
        x['text']) and x['dist'][topic] == max(x['dist'])]
    for i in range(len(data)):
        data[i]["text"] = bad_and_good_word(data[i]["text"], good_words, bad_words)
    return sorted(data, key=lambda x: x['dist'][topic], reverse=True)


def search_by_topic(topic, documents, good_words, bad_words):
    data = [x for x in documents if x['dist'][topic] == max(x['dist'])]
    for i in range(len(data)):
        data[i]["text"] = bad_and_good_word(data[i]["text"], good_words, bad_words)
    return sorted(data, key=lambda x: x['dist'][topic], reverse=True)


def search_by_multiples_words(words, documents, good_words, bad_words, teacher, sigle):
    data = []
    for doc in documents:
        doc_text = join_text(doc['text'])
        if teacher != '' and doc['metadata']['profesor'].lower() != teacher.lower():
            continue

        if sigle != '' and doc['metadata']['sigla'].lower() != sigle.lower():
            continue

        is_here = True
        for word in words:
            if word not in doc_text:
                is_here = False
                break
        if is_here:
            doc["text"] = bad_and_good_word(doc["text"], good_words, bad_words)
            data.append(doc)
    return data
