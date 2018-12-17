from typing import List, Optional, Tuple
from collections import defaultdict
import pickle
import json
from os import path

import click
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, jsonify, request
import numpy as np

from qanta import util
from qanta.dataset import QuizBowlDataset
import nltk


MODEL_PATH = 'tfidf.pickle'
BUZZ_NUM_GUESSES = 5
BUZZ_THRESHOLD = 0.23
W2V_LENGTH = 300
W2V_MODEL = 'full_model.pkl'
W2V_LAMBDA = 0.5
IS_MULTI = True
IS_BERT = True
bert = None
guesser = None

if IS_BERT:
    import qanta_bert
    bert = qanta_bert.qanta_bert()

def get_topk(question, docs):
    arrs = [guesser.answer_docs[d] for d in docs]

    example = bert.predict(question, arrs)

    if len(example) <= 0 or len(example[0]) <= 0:
        return 0
    else:
        return int(example[0][0]["doc_index"])

def guess_and_buzz(model, question_text, idx, multi) -> Tuple[str, bool, int]:
    if multi:
        guesses = model.guess([question_text], [idx], BUZZ_NUM_GUESSES)[0]
    else:
        guesses = model.guess([question_text], [idx], BUZZ_NUM_GUESSES)[0]
    scores = [guess[1] for guess in guesses]
    #print(scores[0] / sum(scores))
    buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
    buzz = buzz and idx > 1
    if multi:
        if not IS_BERT:
            return [guess[0] for guess in guesses], buzz
        else:
            topk = get_topk(question_text, [guess[0] for guess in guesses])
            buzz = scores[topk] / sum(scores) >= BUZZ_THRESHOLD
            return guesses[topk][0], buzz
    else:
        return guesses[0][0], buzz


def batch_guess_and_buzz(model, questions, idxs, multi) -> List[Tuple[str, bool, int]]:
    if multi:
        question_guesses = model.guess(questions, idxs, BUZZ_NUM_GUESSES)
    else:
        question_guesses = model.guess(questions, idxs, BUZZ_NUM_GUESSES)
    outputs = []
    assert(len(questions) == len(question_guesses))
    for i, guesses in tqdm(enumerate(question_guesses)):
        scores = [guess[1] for guess in guesses]
        buzz = scores[0] / sum(scores) >= BUZZ_THRESHOLD
        buzz = buzz and idxs[i] > 1
        #print(scores[0] / sum(scores))
        if multi:
            if not IS_BERT:
                outputs.append(([guess[0] for guess in guesses], buzz))
            else:
                topk = get_topk(questions[i], [guess[0] for guess in guesses])
                buzz = scores[topk] / sum(scores) >= BUZZ_THRESHOLD
                outputs.append((guesses[topk][0], buzz))
        else:
            outputs.append((guesses[0][0], buzz))
    return outputs


class TfidfGuesser:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.i_to_ans = None
        self.w2v = pickle.load(open(W2V_MODEL, "rb"))
        self.answer_docs = defaultdict(str)
        if IS_BERT:
            with open("wiki_lookup.json", "r") as f:
                wiki = json.load(f)
            for k in tqdm(wiki):
                self.answer_docs[k] += ' ' + wiki[k]["text"]

            with open("data/qanta.mapped.2018.04.18.json") as f:
                dataset = json.load(f)
                raw_questions = dataset["questions"]
                GUESSER_TRAIN_FOLD = 'guesstrain'
                BUZZER_TRAIN_FOLD = 'buzztrain'
                TRAIN_FOLDS = {GUESSER_TRAIN_FOLD, BUZZER_TRAIN_FOLD}
                for q in tqdm(raw_questions):
                    if q['fold'] in TRAIN_FOLDS:
                        self.answer_docs[q['page']] += ' ' + q['text']

    def train(self, training_data) -> None:
        questions = training_data[0]
        answers = training_data[1]
        answer_docs = defaultdict(str)
        answer_vecs = defaultdict(lambda: [])
        for q, ans in tqdm(zip(questions, answers)):
            text = ' '.join(q)
            answer_docs[ans] += ' ' + text
            for s in q:
                for w in nltk.word_tokenize(s):
                    if w in self.w2v:
                        answer_vecs[ans].append(self.w2v[w])

        x_array = []
        y_array = []
        self.vecs_array = []
        for ans, doc in tqdm(answer_docs.items()):
            x_array.append(doc)
            y_array.append(ans)
            if(len(answer_vecs[ans]) == 0):
                vec = np.zeros(W2V_LENGTH)
            else:
                vec = np.sum(answer_vecs[ans], axis = 0) / len(answer_vecs[ans])
                vec = vec / np.linalg.norm(vec)
            self.vecs_array.append(vec)

        self.vecs_array = np.array(self.vecs_array)
        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        print("Fitting")
        self.tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 1), min_df=2, max_df=.9
        , stop_words = "english").fit(x_array)
        print("Transform")
        self.tfidf_matrix = self.tfidf_vectorizer.transform(x_array)

    def guess(self, questions: List[str], idxs: Optional[int], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.tfidf_vectorizer.transform(questions)
        guess_matrix = self.tfidf_matrix.dot(representations.T).T
        '''
        print(len(questions), representations.shape)
        print(self.tfidf_matrix.shape)
        print(guess_matrix.shape)
        print(self.vecs_array.shape)
        '''
        vecs_represent = []
        for q in questions:
            temp = []
            for w in nltk.word_tokenize(q):
                if w in self.w2v:
                    temp.append(self.w2v[w])
            if not temp:
                vecs_represent.append(np.zeros(W2V_LENGTH))
            else:
                temp = np.sum(temp, axis = 0) / len(temp)
                temp = temp / np.linalg.norm(temp)
                vecs_represent.append(temp)
        vecs_represent = np.array(vecs_represent)
        self.vecs_array = np.array(self.vecs_array)
        for i in range(len(self.vecs_array)):
            if(type(self.vecs_array[i]) != np.ndarray):
                print(self.vecs_array[i])
        vecs_matrix = self.vecs_array.dot(vecs_represent.T).T
        guess_matrix = guess_matrix.toarray() + W2V_LAMBDA * vecs_matrix
        guess_indices = (-guess_matrix).argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def save(self):
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'vecs_array': self.vecs_array
            }, f)

    @classmethod
    def load(cls):
        with open(MODEL_PATH, 'rb') as f:
            params = pickle.load(f)
            guesser = TfidfGuesser()
            guesser.tfidf_vectorizer = params['tfidf_vectorizer']
            guesser.tfidf_matrix = params['tfidf_matrix']
            guesser.i_to_ans = params['i_to_ans']
            guesser.vecs_array = params['vecs_array']
            return guesser



def create_app(enable_batch=True):
    global guesser
    tfidf_guesser = TfidfGuesser.load()
    guesser = tfidf_guesser
    app = Flask(__name__)

    @app.route('/api/1.0/quizbowl/status', methods=['GET'])
    def status():
        return jsonify({
            'batch': enable_batch,
            'batch_size': 200,
            'ready': True,
            'include_wiki_paragraphs': False
        })

    @app.route('/api/1.0/quizbowl/act', methods=['POST'])
    def act():
        idx = request.json['question_idx']
        question = request.json['text']
        guess, buzz = guess_and_buzz(tfidf_guesser, question, idx, IS_MULTI)
        return jsonify({'guess': guess, 'buzz': True if buzz else False})

    @app.route('/api/1.0/quizbowl/batch_act', methods=['POST'])
    def batch_act():
        idxs = [q['question_idx'] for q in request.json['questions']]
        questions = [q['text'] for q in request.json['questions']]
        return jsonify([
            {'guess': guess, 'buzz': True if buzz else False}
            for guess, buzz in batch_guess_and_buzz(tfidf_guesser, questions, idxs, IS_MULTI)
        ])

    return app


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='0.0.0.0')
@click.option('--port', default=4861)
@click.option('--disable-batch', default=False, is_flag=True)
def web(host, port, disable_batch):
    """
    Start web server wrapping tfidf model
    """
    app = create_app(enable_batch=not disable_batch)
    app.run(host=host, port=port, debug=False)


@cli.command()
def train():
    """
    Train the tfidf model, requires downloaded data and saves to models/
    """
    dataset = QuizBowlDataset(guesser_train=True)
    tfidf_guesser = TfidfGuesser()
    tfidf_guesser.train(dataset.training_data())
    tfidf_guesser.save()


@cli.command()
@click.option('--local-qanta-prefix', default='data/')
@click.option('--retrieve-paragraphs', default=False, is_flag=True)
def download(local_qanta_prefix, retrieve_paragraphs):
    """
    Run once to download qanta data to data/. Runs inside the docker container, but results save to host machine
    """
    util.download(local_qanta_prefix, retrieve_paragraphs)


if __name__ == '__main__':
    cli()
