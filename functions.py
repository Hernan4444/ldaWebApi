import lda
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups


def load_database(option=None):

    if option == 5:
        categ = ['alt.atheism', 'comp.graphics', 'rec.autos', 'rec.sport.hockey', 'sci.med']
    elif option == 10:
        categ = ['alt.atheism', 'comp.graphics', 'comp.windows.x', 'misc.forsale', 'rec.autos',
                 'rec.sport.hockey', 'sci.crypt', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', ]
    else:
        categ = None


    return fetch_20newsgroups(data_home="data/", subset='train', remove=(
        'headers', 'footers', 'quotes'), categories=categ)


def run_lda(newsgroups_train, iterations, alpha, eta, topics):

    tf_vectorizer = CountVectorizer(stop_words='english', max_features=1000, min_df=2, max_df=0.9)
    tf = tf_vectorizer.fit_transform(newsgroups_train.data)

    model = lda.LDA(n_topics=topics, n_iter=iterations, alpha=alpha, eta=eta, refresh=50)
    model.fit(tf)

    topics = []
    for topic_dist in model.topic_word_:
        topics.append(list(zip(
            ["{:0.5f}".format(x) for x in np.array(topic_dist)[np.argsort(topic_dist)][:-20:-1]],
            np.array(tf_vectorizer.get_feature_names())[np.argsort(topic_dist)][:-20:-1]
        )))

    return topics


def run_interactive_lda(newsgroups_train, iterations, alpha, eta, nu, topics, seeds, mode):

    tf_vectorizer = CountVectorizer(stop_words='english', max_features=1000, min_df=2, max_df=0.9)
    tf = tf_vectorizer.fit_transform(newsgroups_train.data)
    vocab = tf_vectorizer.get_feature_names()

    new_seeds = []

    for constrain in seeds:
        new_topic = []
        for word in constrain:
            if word in vocab:
                new_topic.append(vocab.index(word))
        new_seeds.append(new_topic)

    model = lda.LDA(n_topics=topics, n_iter=iterations, alpha=alpha,
                    eta=eta, nu=nu, seed=new_seeds, refresh=100, mode=mode.lower())
                    
    model.fit(tf)

    topics = []
    for topic_dist in model.topic_word_:
        topics.append(list(zip(
            ["{:0.5f}".format(x) for x in np.array(topic_dist)[np.argsort(topic_dist)][:-20:-1]],
            np.array(tf_vectorizer.get_feature_names())[np.argsort(topic_dist)][:-20:-1]
        )))

    return topics
