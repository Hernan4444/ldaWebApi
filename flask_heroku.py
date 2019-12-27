# coding=utf-8
from flask import Flask, request, json, Response, render_template
from json import loads as load_json
import os
import re
from functions import run_lda, load_file, search_by_words, save
from constant import TOKEN
from pathlib import Path
from flask_cors import CORS
from utils import allowed_file, is_xlsx, process_encuestas, process_other_text
from PMI import get_words_pmi
import hashlib
import binascii
from functools import wraps
import pandas as pd

app = Flask(__name__)
CORS(app)


def encript_password(password):
    HASH = hashlib.new('sha256')
    HASH.update(password.encode())
    halt = HASH.digest()
    hash_ = hashlib.pbkdf2_hmac('sha256', halt, b'salt', 1000)
    password_encripty = binascii.hexlify(hash_).decode()
    return password_encripty


def check_token(f):
    @wraps(f)
    def check(*args, **kwargs):
        token = encript_password(request.form.get('token', ''))
        if token != encript_password(TOKEN):
            response = Response(json.dumps({'error': 'No tienes autorización'}),
                                status=401,
                                mimetype='application/json'
                                )
            return response
        return f(*args, **kwargs)
    return check


@app.route("/")
def index():
    with open("index.html", encoding="UTF-8") as file:
        data = "".join(file.readlines())
    return data


@app.route("/pin", methods=["POST", "GET"])
def pin():
    return "pin"


@app.route("/database", methods=["POST", "GET"])
@check_token
def database():
    password = encript_password(request.form.get('password', ''))
    email = request.form.get('email', ' ')
    databases = pd.read_csv('database.tsv',  encoding="UTF-8", sep="\t")
    possibles_databases = databases[(databases.password == password) & (databases.email == email)]
    possibles_databases = possibles_databases[['file_name_client', 'database_name_client']]

    response = Response(json.dumps(possibles_databases.to_dict(orient='records')),
                        status=200,
                        mimetype='application/json'
                        )
    return response


@app.route("/uploadFile", methods=["POST", "GET"])
@check_token
def file_upload():
    file = request.files['file']
    filename = request.form.get('name')
    email = request.form.get('email', ' ')
    password = encript_password(request.form.get('password', ''))

    if file and filename != "" and is_xlsx(filename):
        all_databases = pd.read_csv('database.tsv', encoding="UTF-8", sep="\t")
        file.save(os.path.join("data", filename))
        possible_encuesta_data = all_databases[(all_databases.password == password) & (all_databases.database_name_client == 'EncuestasDocentes')]
        if (possible_encuesta_data.shape[0] == 1):
            exist = True
            exist_filename = possible_encuesta_data.iloc[0].file_name_backend
        else:
            exist = False
            exist_filename = ''

        df_encuestas, indexs = process_encuestas(
            filename, exist=exist, exist_filename=exist_filename)
        save(df_encuestas, 'EncuestasDocentes', email, password, exist, exist_filename, indexs)

    elif file and filename != "" and filename not in os.listdir("data") and allowed_file(filename):
        # Otros archivos no XLS
        file.save(os.path.join("data", filename))
        df_other_text, indexs = process_other_text(filename)
        filename = ".".join(filename.split(".")[:-1])
        save(df_other_text, filename, email, password, False, '', indexs)

    response = Response(json.dumps([]),
                        status=200,
                        mimetype='application/json'
                        )
    return response


@app.route("/searchDocumentMultiplesWords", methods=["POST", "GET"])
@check_token
def search_document_multiples_words():
    words = request.form.get('words', "")
    words = re.compile(' +(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)').split(words)
    words = [x.strip("'").strip('"') for x in words]
    words = [x for x in words if x != ""]
    documents = json.loads(request.form.get('documents', "[]"))
    database = request.form.get('database')
    data, _, indexs = load_file(database)

    news_documents = []
    for client_document, server_document in zip(documents, data.to_dict(orient='records')):
        client_document['text'] = server_document['TEXTO']
        news_documents.append(client_document)

    df = pd.DataFrame(news_documents)

    good_words = [x.lower() for x in json.loads(request.form.get('goodWords', '[]'))]
    bad_words = [x.lower() for x in json.loads(request.form.get('badWords', '[]'))]
    teacher = request.form.get('teacher', '')
    sigle = request.form.get('sigle', '')

    new_documents = search_by_words(words, df, good_words, bad_words, teacher, sigle, indexs)

    response = Response(json.dumps(new_documents),
                        status=200,
                        mimetype='application/json'
                        )

    return response


@app.route("/generateWordCloud", methods=["POST", "GET"])
@check_token
def generate_wordcloud():
    documents = json.loads(request.form.get('documents', "[]"))

    p17 = []
    p18 = []
    for doc in documents:
        doc_text = " ".join([x["text"] for x in doc['text']])

        if doc['metadata']['pregunta'] == 17:
            p17.append(doc_text)
        else:
            p18.append(doc_text)

    p17, p18 = get_words_pmi(p17, p18, True, n_words=40)
    result = {
        "17": {
            "text": [],
            "score": []
        },
        "18": {
            "text": [],
            "score": []
        },

    }
    for word, score in p17:
        result["17"]["text"].append(word)
        result["17"]["score"].append(score)

    for word, score in p18:
        result["18"]["text"].append(word)
        result["18"]["score"].append(score)

    result["18"]["text"] = " ".join(result["18"]["text"])
    result["17"]["text"] = " ".join(result["17"]["text"])
    response = Response(json.dumps(result),
                        status=200,
                        mimetype='application/json'
                        )

    return response


@app.route("/lda", methods=["POST", "GET"])
@check_token
def apply_lda():

    iterations = int(request.form.get('iterations'))
    mode = request.form.get('mode', None)
    alpha = float(request.form.get('alpha', "0").replace(",", "."))
    beta = float(request.form.get('beta', "0").replace(",", "."))
    topics = int(request.form.get('topics'))
    database = request.form.get('database')
    nu = float(request.form.get('nu', "0").replace(",", "."))
    seed = json.loads(request.form.get('seeds', "[]"))

    stopwords_spanish = request.form.get('stopwords', "False") == "True"
    steeming = request.form.get('steeming', "False") == "True"
    data, is_encuesta, _ = load_file(database)

    mode = mode if mode != "LDA" else None
    result = run_lda(data, iterations, alpha, beta, topics, is_encuesta,
                     stopwords_spanish, steeming, nu, seed, mode)

    response = Response(json.dumps(result),
                        status=200,
                        mimetype='application/json'
                        )
    return response


if __name__ == "__main__":
    app.run()
