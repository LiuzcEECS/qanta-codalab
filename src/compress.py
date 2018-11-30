import os
ans = {}
input_list = ["../data/qanta.dev.2018.04.18.json", "../data/qanta.test.2018.04.18.json",
        "../data/qanta.train.2018.04.18.json", "../data/qanta.mapped.2018.04.18.json"]
import gensim
import json
import nltk

w2v = gensim.models.KeyedVectors.load_word2vec_format("full_model.bin", binary=True)

for name in input_list:
    with open(name, "r") as f:
        questions = json.load(f)['questions']
    for q in questions:
        for s in nltk.sent_tokenize(q['text']):
            for w in nltk.word_tokenize(s):
                if w in w2v:
                    ans[w] = w2v[w]
import pickle
pickle.dump(ans, open("full_model.pkl", "wb"))
