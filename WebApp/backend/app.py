from flask import Flask
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from api.ReqHandler import ReqHandler

app = Flask(__name__, static_url_path = '', static_folder = 'frontend/build')
CORS(app)
api = Api(app)

@app.route('/', defaults={'path':''})
def serve(path):
    return send_from_directory(app.static_folder, 'index.html')

api.add_resource(ReqHandler, '/')