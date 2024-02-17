from flask import Flask, render_template
from flask_restful import reqparse, Api
from werkzeug.exceptions import BadRequest
from flask import jsonify
import flask

import numpy as np
import pandas as pd
import ast

import os
import json

from model import predict_yield

curr_path = os.path.dirname(os.path.realpath(__file__))

feature_cols = ['AverageRainingDays', 'clonesize', 'AverageOfLowerTRange',
    'AverageOfUpperTRange', 'honeybee', 'osmia', 'bumbles', 'andrena']

context_dict = {
    'feats': feature_cols,
}

app = Flask(__name__)
api = Api(app)

# # FOR FORM PARSING
parser = reqparse.RequestParser()
parser.add_argument('list', type=list, location='json', required=True, help="List cannot be blank!")

@app.errorhandler(BadRequest)
def handle_bad_request(error):
    response = jsonify({'message': error.data['message']})
    response.status_code = 400
    return response

@app.route('/api/predict', methods=['GET','POST'])
def api_predict():
    data = flask.request.form.get('single input')
    
    # converts json to int 
    i = ast.literal_eval(data)
    
    y_pred = predict_yield(np.array(i).reshape(1,-1))
    
    return {'message':"success", "pred":json.dumps(int(y_pred))}

@app.route('/predict', methods=['POST'])
def predict():
    # flask.request.form.keys() will print all the input from form
    test_data = []
    for val in flask.request.form.values():
        test_data.append(float(val))
    test_data = np.array(test_data).reshape(1,-1)

    y_pred = predict_yield(test_data)
    context_dict['pred']= y_pred

    print(y_pred)

    return render_template('index.html', **context_dict)

@app.route('/api/predict/json', methods=['POST'])
def api_predict_json():
    data = parser.parse_args()
    input_list = data['list']  
    print("data : ",input_list)
    
    y_pred = predict_yield(np.array(input_list).reshape(1,-1))
    
    return {'message':"success", "pred":json.dumps(int(y_pred))}

@app.route('/api/predict_batch', methods=['POST'])
def api_predict_batch():
    data = flask.request.get_json()
    inputs = data['inputs']

    if not all(isinstance(item, list) for item in inputs):
        return {"message": "Invalid input format, expected a list of lists."}, 400

    predictions = []

    for input_list in inputs:
        input_array = np.array(input_list).reshape(1, -1)
        y_pred = predict_yield(input_array)
        predictions.append(int(y_pred))

    return {'message': "success", "predictions": json.dumps(predictions)}


@app.route('/')
def index():
    
    # render the index.html templete
    
    return render_template("index.html", **context_dict)



if __name__ == "__main__":
    app.run()