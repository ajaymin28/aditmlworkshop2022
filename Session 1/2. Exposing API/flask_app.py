# -*- coding: utf-8 -*-
"""

"""

from flask import Flask, Blueprint
from flask_restful import Api as restful_api
from flask_restx import Api
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = 'asfnsdlgk23242jk4n1qwe2jk4n'


blueprint = Blueprint('api', __name__)
swagger_api = Api(blueprint,version='1.0', title='ModelInference', description='Model Inferecne', doc='/apidoc/', prefix='/api')
app.register_blueprint(blueprint)
api = restful_api(app)
