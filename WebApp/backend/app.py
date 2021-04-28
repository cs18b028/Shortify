######################################################################################################
#                                                                                                    #
#   Flask App                                                                                        #
#                                                                                                    #
#   endpoints : serve (/), question (/question), summary (/summary)                                  #
#                                                                                                    #
######################################################################################################

# importing the necessary python libraries and modules

from flask import Flask, request, json, jsonify
from flask_cors import CORS, cross_origin
from core.process_data import process_text
from core.relevant_questions import get_rel_que
from core.summarizer import summarizer

app = Flask(__name__)
cors = CORS(app)

# serve

@app.route('/')
@cross_origin()
def serve():
    return 'This the backend of Shortify-Webapp'

# questions

@app.route('/question', methods=['GET', 'POST'])
@cross_origin()
def question():
    if request.method == 'POST':
        return 'questions post'
    else:
        print("getting questions")
        params = request.json
        if (params == None):
            params = request.args
        print(params)
        query = params['query']
        query = process_text(query)
        results = get_rel_que(query)
        return jsonify({'results': results})

# summaries

@app.route('/summary', methods=['GET', 'POST'])
@cross_origin()
def summary():
    if request.method == 'POST':
        return 'summaries post'
    else:
        print("getting summaries")
        params = request.json
        if (params == None):
            params = request.args
        print(params)
        query = params['query']
        query = process_text(query)
        results = summarizer(query)
        return jsonify({'results': results})

# main

if __name__ == '__main__':
    app.debug = True
    app.run()