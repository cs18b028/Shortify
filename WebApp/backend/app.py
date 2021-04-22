from flask import Flask, request, json, jsonify
from flask_cors import CORS, cross_origin
from core.process_data import process_text
from core.summarizer import summarizer

app = Flask(__name__)
cors = CORS(app)

'''@app.route('/')
@cross_origin()
def serve():
    return 'This the backend of Shortify-Webapp'
'''

@app.route('/api')
@cross_origin()
def result():
    params = request.json
    if (params == None):
        params = request.args
    query = params["query"]
    query = process_text(query)
    results = summarizer(query)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.debug = True
    app.run()