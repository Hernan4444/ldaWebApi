import os
import re

from collections import Counter
from functools import reduce


import numpy as np
import nltk

from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer
from nltk.stem import WordNetLemmatizer

from gensim.parsing.preprocessing import strip_punctuation
from gensim.corpora import Dictionary

nltk.download('stopwords')
nltk.download('wordnet')  # download wordnet to be used in lemmatization

sw = stopwords.words("spanish")
stemmer = SpanishStemmer(ignore_stopwords=True)


def preprocess(texts, stem=True):
    # tokenization
    texts = [re.findall(r'\w+', line.lower()) for line in texts]
    # remove stopwords
    texts = [[word for word in line if word not in sw] for line in texts]
    # remove punctuation
    texts = [strip_punctuation(' '.join(line)).split() for line in texts]
    # remove words that are only 1-2 character
    texts = [[token for token in line if len(token) > 2] for line in texts]
    # remove numbers
    texts = [[token for token in line if not token.isnumeric()]
             for line in texts]

    # stemming
    stem_dict = dict()
    
    if stem:
        stemmed = [[stemmer.stem(word) for word in line] for line in texts]

        for line in texts:
            for word in line:
                stem = stemmer.stem(word)
                if stem_dict.get(stem, None) == None:
                    stem_dict[stem] = []
                stem_dict[stem].append(word)
        for stem in stem_dict:
            stem_dict[stem] = Counter(stem_dict[stem]).most_common(1)[0][0]
#    stem_dict = {stemmer.stem(word): word  for line in texts for word in line}

    return stemmed, stem_dict


class PMI:

    def __init__(self):
        self.total_counts = Counter()
        self.contexts_counts = []

    def fit(self, *contexts):
        """
        contexts: corpuses
        """
        for context in contexts:
            ctx_count = Counter()
            for doc in context:
                ctx_count += Counter(dict(doc))
            self.contexts_counts.append(ctx_count)
            self.total_counts += ctx_count

    def ppmi(self, word, ctx):
        """
        word: word
        ctx: index of context
        """
        p_word_given_ctx = self.contexts_counts[ctx].get(
            word, 0) / np.sum(list(self.contexts_counts[ctx].values()))
        p_word = self.total_counts.get(
            word, 0) / np.sum(list(self.total_counts.values()))
        return max(np.log2(p_word_given_ctx / p_word), 0)

    def get_top_words(self, n, ctx, min_prop=0.005):
        """
        n: Amount of top words
        ctx: index of ctx
        """
        total = np.sum(list(self.contexts_counts[ctx].values()))
        proportions = np.array(
            [self.contexts_counts[ctx][i]/total for i in self.contexts_counts[ctx].keys()])
#        print(proportions.max(), proportions.min())
        ranking = np.array([[self.ppmi(i, ctx), i]
                            for i in self.contexts_counts[ctx].keys()])
        ranking = list(ranking[proportions > min_prop])
        ranking.sort(reverse=True, key=lambda x: x[0])
        return ranking[:min(n, len(ranking))]


def get_words_pmi(documents_1, documents_2, stem, n_words=20):
    """
    documents_1: list of strings
    documents_2: list of strings
    stem: bool
    """
    processed_1, stem_dict_1 = preprocess(documents_1, stem)
    processed_2, stem_dict_2 = preprocess(documents_2, stem)
    dictionary_1 = Dictionary(processed_1)
    dictionary_2 = Dictionary(processed_2)
    corpus_1 = [dictionary_1.doc2bow(text) for text in processed_1]
    corpus_2 = [dictionary_2.doc2bow(text) for text in processed_2]

    pmi = PMI()
    pmi.fit(corpus_1, corpus_2)
    if stem:
        pmi_1 = list(map(lambda x: (
            stem_dict_1[dictionary_1.get(x[1])], x[0]), pmi.get_top_words(n_words, 0)))
        pmi_2 = list(map(lambda x: (
            stem_dict_2[dictionary_2.get(x[1])], x[0]), pmi.get_top_words(n_words, 1)))
    else:
        pmi_1 = list(map(lambda x: (dictionary_1.get(
            x[1]), x[0]), pmi.get_top_words(n_words, 0)))
        pmi_2 = list(map(lambda x: (dictionary_2.get(
            x[1]), x[0]), pmi.get_top_words(n_words, 1)))
    return pmi_1, pmi_2


def get_bigrams_pmi(documents_1, documents_2, stem, n_words=20):

    def get_bigrams(documents, sep="-"):
        bigrams = [
            list(map(lambda x: "{}{}{}".format(x[0], sep, x[1]), ngrams(doc, 2))) for doc in documents]
        return bigrams

    def bigram_to_str(bigram, dictionary, sep="-"):
        stems = bigram.split(sep)
        if stem:
            return "{}-{}".format(dictionary[stems[0]], dictionary[stems[1]])
        else:
            return "{}-{}".format(stems[0], stems[1])

    processed_1, stem_dict_1 = preprocess(documents_1, stem)
    processed_2, stem_dict_2 = preprocess(documents_2, stem)

    bigrams_1 = get_bigrams(processed_1)
    bigrams_2 = get_bigrams(processed_2)
    dictionary_1 = Dictionary(bigrams_1)
    dictionary_2 = Dictionary(bigrams_2)
    corpus_1 = [dictionary_1.doc2bow(text) for text in bigrams_1]
    corpus_2 = [dictionary_2.doc2bow(text) for text in bigrams_2]
    pmi = PMI()
    pmi.fit(corpus_1, corpus_2)
    pmi_1 = list(map(lambda x: (bigram_to_str(dictionary_1.get(
        x[1]), stem_dict_1), x[0]), pmi.get_top_words(30, 0, min_prop=0.0003)))
    pmi_2 = list(map(lambda x: (bigram_to_str(dictionary_2.get(
        x[1]), stem_dict_2), x[0]), pmi.get_top_words(30, 1, min_prop=0.0003)))
    return pmi_1, pmi_2


if __name__ == "__main__":
    import pandas as pd
    p17 = pd.read_csv('data/P17.tsv', sep="\t",
                      header=None)  # Comentarios positivos
    p18 = pd.read_csv('data/P18.tsv', sep="\t",
                      header=None)  # Comentarios por mejorar
    inquiries = p17.append(p18)
    inquiries = inquiries.loc[pd.notnull(inquiries[3])]
    inquiries = inquiries.reset_index(drop=True)
    inquiries = inquiries.groupby([1])[3].apply(
        lambda x: " ".join(x)).to_frame()
    p17 = p17.groupby([1])[3].apply(lambda x: " ".join(x)).to_frame()
    p18 = p18.groupby([1])[3].apply(lambda x: " ".join(x)).to_frame()
    # p17 = ["asdads sakdjad aslkdja sldkjasl d", "123 sakdjad aslkdja sldkjasl d", "asdads sakdjad aslkdja sldkjasl d"]
    # p18 = ["12312 2131sad", "salkdja0921 assako", "slakdn23jq0912jios"]
    # words_pmi = get_words_pmi(p17, p18, True, n_words=40)
    # print(words_pmi)
    bigrams_pmi = get_bigrams_pmi(p17[3], p18[3], True, n_words=30)
    print(bigrams_pmi)
