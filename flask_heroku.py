# coding=utf-8
from flask import Flask, request, json, Response
from json import loads as load_json
import os
import re
from functions import load_database, run_lda, run_interactive_lda
from pathlib import Path  # python3 only
from dotenv import load_dotenv
from flask_cors import CORS

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=".env")


TOKEN = os.getenv("token")
print(TOKEN)
app = Flask(__name__)
CORS(app)


@app.route("/pin", methods=["POST", "GET"])
def pin():
    load_database()
    return "pin"


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

    if database == "NewGroups.5":
        data = load_database(5)
    elif database == "NewGroups.10":
        data = load_database(10)
    else:
        data = load_database()

    if mode == "LDA":
        result = run_lda(data, iterations, alpha, beta, topics)
    else:
        nu = float(request.form.get('nu', "0").replace(",", "."))
        seed = json.loads(request.form.get('seeds', "[]"))
        result = run_interactive_lda(data, iterations, alpha, beta, nu, topics, seed, mode)

    # return string_r
    response = Response(json.dumps(result),
                        status=200,
                        mimetype='application/json'
                        )
    return response


if __name__ == "__main__":
    app.run()
