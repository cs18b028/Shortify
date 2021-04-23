from flask import Flask, request, json, jsonify
from flask_cors import CORS, cross_origin
from core.process_data import process_text
from core.relevant_questions import get_rel_que
from core.summarizer import summarizer

app = Flask(__name__)
cors = CORS(app)

@app.route('/')
@cross_origin()
def serve():
    return 'This the backend of Shortify-Webapp'

@app.route('/question', methods=['GET', 'POST'])
@cross_origin()
def que():
    if request.method == 'POST':
        return 'hello'
    else:
        print("hello")
        params = request.json
        if (params == None):
            params = request.args
        print(params)
        query = params['query']
        query = process_text(query)
        results = get_rel_que(query)
        return jsonify({'results': results})


@app.route('/summary', methods=['GET', 'POST'])
@cross_origin()
def summary():
    if request.method == 'POST':
        return 'hello'
    else:
        print("hello")
        params = request.json
        if (params == None):
            params = request.args
        print(params)
        query = params['query']
        query = process_text(query)
        results = summarizer(query)
        return jsonify({'results': results})

if __name__ == '__main__':
    app.debug = True
    app.run()