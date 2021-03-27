from flask import Flask
#An extension for flask that adds support for quickly building REST APIs
from flask_restful import Api, Resource, reqparse
from api.ReqHandler import ReqHandler
from flask_cors import CORS

app = Flask(__name__, static_url_path='', static_folder='./frontend/build')
CORS(app)
api = Api(app)

@app.route('/',defaults={'path':''})
def serve(path):
    return send_from_directory(app.static_folder, 'index.html')

api.add_resource(ReqHandler, '/')