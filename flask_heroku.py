# coding=utf-8
from flask import Flask, request, json, Response, render_template
from json import loads as load_json
import os
import re
from functions import load_database, run_lda, run_interactive_lda, load_file
from pathlib import Path  # python3 only
from dotenv import load_dotenv
from flask_cors import CORS

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=".env")


TOKEN = os.getenv("token")

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = set(['tsv', 'csv', 'txt'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    with open("page/index.html", encoding="UTF-8") as file:
        data = "".join(file.readlines())
    return data


@app.route("/pin", methods=["POST", "GET"])
def pin():
    load_database()
    return "pin"


@app.route("/database", methods=["POST", "GET"])
def database():
    files = [
        ["NewGroups.5", "NewGroups (5 categorías)"],
        ["NewGroups.10", "NewGroups (10 categorías)"],
    ]

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

    if file and filename != "" and filename not in os.listdir("data") and allowed_file(filename):
        file.save(os.path.join("data", filename))

        with open("database.json", encoding="UTF-8") as file:
            data = json.load(file)

        index = data["index"] + 1
        data["files"].append(["file-{}".format(index), filename.rsplit('.', 1)[0], filename])
        data["index"] += 1

        with open("database.json", encoding="UTF-8", mode="w") as file:
            json.dump(data, file)

    files = [
        ["NewGroups.5", "NewGroups (5 categorías)"],
        ["NewGroups.10", "NewGroups (10 categorías)"],
    ]

    with open("database.json", encoding="UTF-8") as file:
        data = json.load(file)

    files.extend(data["files"])

    response = Response(json.dumps(files),
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

    stop_words = False
    if database == "NewGroups.5":
        data = load_database(5)
        stop_words = True
    elif database == "NewGroups.10":
        data = load_database(10)
        stop_words = True
    else:
        with open("database.json", encoding="UTF-8") as file:
            data = load_file(database, json.load(file))

    if mode == "LDA":
        result = run_lda(data, iterations, alpha, beta, topics, stop_words)
    else:
        nu = float(request.form.get('nu', "0").replace(",", "."))
        seed = json.loads(request.form.get('seeds', "[]"))
        result = run_interactive_lda(data, iterations, alpha, beta, nu,
                                     topics, seed, mode, stop_words)

    # return string_r
    response = Response(json.dumps(result),
                        status=200,
                        mimetype='application/json'
                        )
    return response


if __name__ == "__main__":
    app.run()
