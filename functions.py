import lda
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
import json
from string import punctuation, ascii_uppercase
from collections import defaultdict


STEMMER = SnowballStemmer('spanish')
NON_LETTERS = list(punctuation)

# we add spanish punctuation
NON_LETTERS.extend(['¿', '¡', '-'])
NON_LETTERS.extend(map(str, range(10)))


def filter_df(df, words, indexs):
    if len(words) == 0:
        return df
    indexs_doc = None
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


def generate_secret_code(number):
    first_number, second_number = random.randint(10, 99), random.randint(10, 99)
    id_ = number + first_number + second_number
    first_letters = "".join(random.sample(ascii_uppercase, 2))
    second_letters = "".join(random.sample(ascii_uppercase, 2))
    code = "{}{}{}{}{}".format(first_number, first_letters, id_, second_letters, second_number)
    return code


def save(df, filename, email, password, exist, exist_filename, indexs):
    if df is None:
        return

    if exist:
        new_filename = exist_filename
        indexs_filename = exist_filename + ".idx"
    else:
        all_databases = pd.read_csv('database.tsv', encoding="UTF-8", sep="\t")
        index = all_databases.shape[0]  # numbers of rows
        code = generate_secret_code(index)
        new_filename = 'file_{}'.format(index)
        indexs_filename = 'file_{}.idx'.format(index)

    df.to_csv(os.path.join("data", new_filename), sep="\t")
    with open(os.path.join("data", indexs_filename), "w", encoding="UTF-8") as file:
        json.dump(indexs, file)

    if not exist:
        with open('database.tsv', "a") as file:
            file.write("{}\t{}\t{}\t{}\t{}\n".format(code, filename, new_filename, password, email))


def load_file(filename):
    df_dataset = pd.read_csv('database.tsv', encoding="UTF-8", sep="\t")
    df_data = df_dataset[df_dataset.file_name_client == filename].iloc[0]
    is_encuesta = df_data.database_name_client == "EncuestasDocentes"
    df = pd.read_csv(os.path.join("data", df_data.file_name_backend),
                     encoding="UTF-8", sep="\t", index_col=0)
    with open(os.path.join("data", df_data.file_name_backend + ".idx"), encoding="UTF-8") as file:
        idx = json.load(file)

    return df, is_encuesta, idx


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

    doc_topic = []
    teachers_options = set()
    sigles_options = set()
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
        doc_topic.append(data)
        teachers_options.add(doc["NOMBRE_DEL_DOCENTE"])
        sigles_options.add(doc["SIGLA"])

    return {
        "word_topic": word_topics,
        "doc_topic": doc_topic,
        "filter": {
            'teacher': list(teachers_options),
            'sigle': list(sigles_options)
        }
    }


def process_text(doc):
    data = []
    new_doc = doc.replace("\n", " ")
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


def search_by_words(words, df, good_words, bad_words, teacher, sigle, index):
    data = []
    documents = filter_df(df, words, index)
    good_words = [STEMMER.stem(word).lower() for word in good_words]
    bad_words = [STEMMER.stem(word).lower() for word in bad_words]
    mark_words_index, word_types = get_index(good_words, bad_words, index, documents.index)

    for index, doc in documents.iterrows():
        doc = doc.to_dict()

        if teacher != '' and doc['metadata']['profesor'].lower() != teacher.lower():
            continue

        if sigle != '' and doc['metadata']['sigla'].lower() != sigle.lower():
            continue

        doc["text"] = bad_and_good_word(index, doc["text"], mark_words_index, word_types)
        data.append(doc)

    return data
