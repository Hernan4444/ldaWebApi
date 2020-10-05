# coding=utf-8
from flask import Flask, request, json, Response, render_template
from json import loads as load_json
import os
from os.path import join
import re
from functions import run_lda, search_by_words
from constant import TOKEN
from pathlib import Path
from flask_cors import CORS
from utils import allowed_file, is_xlsx, encript_password, generate_filename
from preprocess_file import process_encuestas, process_other_text
from PMI import get_words_pmi
import hashlib
import binascii
from functools import wraps
import pandas as pd
from database import Database

app = Flask(__name__)
CORS(app)
DATABASE = Database()


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
    return Response("PIN",status=200)


@app.route("/database", methods=["POST", "GET"])
@check_token
def database():
    password = encript_password(request.form.get('password', ''))
    df = DATABASE.get_databases(password)

    response = Response(json.dumps(df.to_dict(orient='records')),
                        status=200,
                        mimetype='application/json'
                        )
    return response


@app.route("/delete", methods=["POST", "GET"])
@check_token
def delete():
    name_client = request.form.get('name')
    password = encript_password(request.form.get('password', ''))
    text = "Base de datos no encontrada"
    status = "error"

    if DATABASE.remove(name_client, password):
        text = "Base de datos eliminada con éxito"
        status = "successfull"

    response = Response(json.dumps({"status": status, "message": text}),
                        status=200,
                        mimetype='application/json'
                        )
    return response


@app.route("/uploadFile", methods=["POST", "GET"])
@check_token
def file_upload():
    file = request.files['file']
    original_name = request.form.get('originalName')
    name_client = request.form.get('name')
    password = encript_password(request.form.get('password', ''))
    filename = generate_filename()

    text = "Base de datos subida con éxito"
    status = "successfull"
    if is_xlsx(original_name):
        filename += ".xlsx"
        file.save(os.path.join("data", filename))
        df = DATABASE.found(name_client, password)

        if df.shape[0] == 1:
            exist_filename = df.iloc[0].file_name_backend
            text = "Base de datos concatenada a la anterior con éxito"
        else:
            exist_filename = ""

        try:
            df_encuestas, indexs = process_encuestas(filename, exist_filename)
            if df_encuestas is None:
                status = "error"
                text = "El archivo no posee algunas de las siguientes columnas: ANO_APLICACION, PERIODO_APLICACION, NOMBRE_UA_CURSO, NOMBRE_ASIGNATURA, SIGLA, SECCION, NOMBRE_DEL_DOCENTE"
            else:
                DATABASE.save(df_encuestas, name_client, password, exist_filename, indexs, True)
                
        except Exception as e:
            status = "error"
            text = "Error con el archivo: {}".format(e)

    elif allowed_file(original_name):
        # Otros archivos no XLS
        file.save(os.path.join("data", filename))
        df = DATABASE.found(name_client, password)

        if df.shape[0] == 1:
            exist_filename = df.iloc[0].file_name_backend
            text = "Base de datos concatenada a la anterior con éxito"
        else:
            exist_filename = ""
        
        try:
            df_other_text, indexs = process_other_text(filename, exist_filename)
            DATABASE.save(df_other_text, name_client, password, exist_filename, indexs, False)
                
        except Exception as e:
            status = "error"
            text = "Error con el archivo: {}".format(e)

    response = Response(json.dumps({"status": status, "message": text}),
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
    
    wordsOr = request.form.get('wordsOr', "")
    wordsOr = re.compile(' +(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)').split(wordsOr)
    wordsOr = [x.strip("'").strip('"') for x in wordsOr]
    wordsOr = [x for x in wordsOr if x != ""]

    documents = json.loads(request.form.get('documents', "[]"))
    database = request.form.get('database')
    data, indexs, _ = DATABASE.load_file(database)

    news_documents = []
    for client_document, server_document in zip(documents, data.to_dict(orient='records')):
        client_document['text'] = server_document['TEXTO']
        news_documents.append(client_document)

    df = pd.DataFrame(news_documents)

    good_words = [x.lower() for x in json.loads(request.form.get('goodWords', '[]'))]
    bad_words = [x.lower() for x in json.loads(request.form.get('badWords', '[]'))]
    teacher = request.form.get('teacher', '')
    sigle = request.form.get('sigle', '')
    is_teacher_pool = request.form.get('is_teacher_pool', '') == "true" 
    new_documents = search_by_words(words, wordsOr, df, good_words, bad_words, teacher, sigle, indexs, is_teacher_pool)

    response = Response(json.dumps(new_documents),
                        status=200,
                        mimetype='application/json'
                        )

    return response


@app.route("/generateWordCloud", methods=["POST", "GET"])
@check_token
def generate_wordcloud():
    documents = json.loads(request.form.get('documents', "[]"))
    p17, p18 = [], []
    for doc in documents:
        doc_text = " ".join([x["text"] for x in doc['text']])
        if doc['metadata']['pregunta'] == 17:
            p17.append(doc_text)
        else:
            p18.append(doc_text)

    p17, p18 = get_words_pmi(p17, p18, True, n_words=40)
    result = {
        "17": {"text": [], "score": []},
        "18": {"text": [], "score": []},
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
    try:
        data, _, is_encuesta = DATABASE.load_file(database)
    except IndexError:
        response = Response(json.dumps({"result": [], "status": "error", "message": "La base de datos no existe"}),
                            status=200,
                            mimetype='application/json'
                            )
        return response                  
    
    except Exception as e:
        print(e)
        response = Response(json.dumps({"result": [], "status": "error", "message": "Hubo un error con esta base de datos."}),
                            status=200,
                            mimetype='application/json'
                            )
        return response                  

    mode = mode if mode != "lda" else None
    try:
        result = run_lda(data, iterations, alpha, beta, topics, is_encuesta,
                        stopwords_spanish, steeming, nu, seed, mode)
    except Exception as e:
        print(e)
        response = Response(json.dumps({"result": [], "status": "error", "message": "Hubo un error con esta base de datos."}),
                            status=200,
                            mimetype='application/json'
                            )
        return response 

    response = Response(json.dumps(result),
                        status=200,
                        mimetype='application/json'
                        )
    return response


if __name__ == "__main__":
    app.run()
