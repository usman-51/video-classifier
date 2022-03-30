from os import path
from flask_restful import Resource, Api, reqparse
from flask import Flask, request
from flask_cors import CORS
from flask import jsonify
from video_classifier import *

import pandas as pd
import numpy as np
import random
import time

app = Flask(__name__)
CORS(app)

app.config['DEBUG'] = True


api = Api(app)



class video_classifier(Resource):
    def post(self):
        try:
            path=request.form['path']
            test_frames,pred = sequence_prediction(path,model)
            return pred,200
        except Exception as e:
            return {"Error occured":str(e)}, 400



api.add_resource(video_classifier, '/get_video_classifier')

if __name__ == "__main__":
    app.run(debug=True,port=8099)