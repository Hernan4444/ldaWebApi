# coding=utf-8
from flask import Flask, request, json, Response, render_template
from json import loads as load_json
import os
import re
from functions import load_database, run_lda, run_interactive_lda, load_file, \
    search_by_words, search_by_word_and_topic, search_by_topic, search_by_multiples_words
from constant import TOKEN
from pathlib import Path  # python3 only
# from dotenv import load_dotenv
from flask_cors import CORS
from utils import allowed_file, is_xlsx, process_encuestas
from PMI import get_words_pmi

# env_path = Path('.') / '.env'
# load_dotenv(dotenv_path=".env")


app = Flask(__name__)
CORS(app)


@app.route("/page")
def index_page():
    with open("page/index.html", encoding="UTF-8") as file:
        data = "".join(file.readlines())
    return data

@app.route("/")
def index():
    with open("index.html", encoding="UTF-8") as file:
        data = "".join(file.readlines())
    return data


@app.route("/pin", methods=["POST", "GET"])
def pin():
    return "pin"


@app.route("/database", methods=["POST", "GET"])
def database():
    files = []

    with open("database.json", encoding="UTF-8") as file:
        data = json.load(file)

    files.extend(data["files"])

    response = Response(json.dumps(files),
                        status=200,
                        mimetype='application/json'
                        )
    return response


@app.route("/uploadFile", methods=["POST", "GET"])
def file_upload():
    token = request.form.get('token')
    if token != TOKEN:

        response = Response(json.dumps({'error': 'No tienes autorización'}),
                            status=401,
                            mimetype='application/json'
                            )
        return response

    file = request.files['file']
    filename = request.form.get('name')
    if file and filename != "" and is_xlsx(filename):
        file.save(os.path.join("data", filename))

        title, exist = process_encuestas(filename)

        if not exist and title != "ERROR":

            with open("database.json", encoding="UTF-8") as file:
                data = json.load(file)

            if len(data["files"]) == 0:
                index = 0
            else:
                index = int(data["files"][-1][0].split("-")[1]) + 1
            data["files"].append(["file-{}".format(index), title.rsplit('.', 1)[0], title])

            with open("database.json", encoding="UTF-8", mode="w") as file:
                json.dump(data, file, indent=2)

    elif file and filename != "" and filename not in os.listdir("data") and allowed_file(filename):
        file.save(os.path.join("data", filename))

        with open("database.json", encoding="UTF-8") as file:
            data = json.load(file)

        if len(data["files"]) == 0:
            index = 0
        else:
            index = int(data["files"][-1][0].split("-")[1]) + 1

        data["files"].append(["file-{}".format(index), filename.rsplit('.', 1)[0], filename])

        with open("database.json", encoding="UTF-8", mode="w") as file:
            json.dump(data, file, indent=2)

    files = []

    with open("database.json", encoding="UTF-8") as file:
        data = json.load(file)

    files.extend(data["files"])

    response = Response(json.dumps(files),
                        status=200,
                        mimetype='application/json'
                        )
    return response


@app.route("/searchDocumentMultiplesWords", methods=["POST", "GET"])
def search_document_multiples_words():
    token = request.form.get('token')
    if token != TOKEN:

        response = Response(json.dumps({'error': 'No tienes autorización'}),
                            status=401,
                            mimetype='application/json'
                            )
        return response

    words = request.form.get('words', "")
    words = re.compile(' +(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)').split(words)
    words = [x.strip("'").strip('"') for x in words]
    documents = json.loads(request.form.get('documents', "[]"))
    good_words = [x.lower() for x in json.loads(request.form.get('goodWords', '[]'))]
    bad_words = [x.lower() for x in json.loads(request.form.get('badWords', '[]'))]
    teacher = request.form.get('teacher', '')
    sigle = request.form.get('sigle', '')


    new_documents = search_by_multiples_words(words, documents, good_words, bad_words, teacher, sigle)

    response = Response(json.dumps(new_documents),
                        status=200,
                        mimetype='application/json'
                        )
                        
    return response


@app.route("/generateWordCloud", methods=["POST", "GET"])
def generate_wordcloud():
    token = request.form.get('token')
    if token != TOKEN:

        response = Response(json.dumps({'error': 'No tienes autorización'}),
                            status=401,
                            mimetype='application/json'
                            )
        return response

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
        "17":{
        "text":[],
        "score":[]
        },
        "18":{
        "text":[],
        "score":[]
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

@app.route("/searchDocument", methods=["POST", "GET"])
def search_document():
    token = request.form.get('token')
    if token != TOKEN:

        response = Response(json.dumps({'error': 'No tienes autorización'}),
                            status=401,
                            mimetype='application/json'
                            )
        return response

    word = request.form.get('word', "")
    topic = int(request.form.get('topic'))
    mode = request.form.get('mode')
    documents = json.loads(request.form.get('documents', "[]"))
    good_words = json.loads(request.form.get('goodWords', '["domina", "interactiva"]'))
    bad_words = json.loads(request.form.get('badWords', '["valoro", "disponibilidad"]'))

    if mode == "word":
        new_documents = search_by_words(word, topic, documents, good_words, bad_words)
    elif mode == "topic":
        new_documents = search_by_topic(topic, documents, good_words, bad_words)
    else:
        new_documents = search_by_word_and_topic(word, topic, documents, good_words, bad_words)

    response = Response(json.dumps(new_documents),
                        status=200,
                        mimetype='application/json'
                        )
                        
    return response


@app.route("/lda", methods=["POST", "GET"])
def test():
    token = request.form.get('token')
    if token != TOKEN:

        response = Response(json.dumps({'error': 'No tienes autorización'}),
                            status=401,
                            mimetype='application/json'
                            )
        return response

    iterations = int(request.form.get('iterations'))
    mode = request.form.get('mode')
    alpha = float(request.form.get('alpha', "0").replace(",", "."))
    beta = float(request.form.get('beta', "0").replace(",", "."))
    topics = int(request.form.get('topics'))
    database = request.form.get('database')
    
    stop_words_spanish = request.form.get('stopwords', "False") == "True"
    stop_words_array = []
    if stop_words_spanish:
        with open("stopwords_spanish.txt", encoding="UTF-8") as file:
            for line in file:
                stop_words_array.append(line.strip())

    steeming = request.form.get('steeming', "False") == "True"

    english_stopwords = False
    if database == "NewGroups.5":
        data = load_database(5)
        english_stopwords = True
    elif database == "NewGroups.10":
        data = load_database(10)
        english_stopwords = True
    else:
        with open("database.json", encoding="UTF-8") as file:
            data = load_file(database, json.load(file))

    if mode == "LDA":
        result = run_lda(data, iterations, alpha, beta, topics,
                         english_stopwords, stop_words_array, steeming)
    else:
        nu = float(request.form.get('nu', "0").replace(",", "."))
        seed = json.loads(request.form.get('seeds', "[]"))
        result = run_interactive_lda(data, iterations, alpha, beta, nu,
                                     topics, seed, mode, english_stopwords, stop_words_array, steeming)

    # return string_r
    response = Response(json.dumps(result),
                        status=200,
                        mimetype='application/json'
                        )
    return response


if __name__ == "__main__":
    app.run()
