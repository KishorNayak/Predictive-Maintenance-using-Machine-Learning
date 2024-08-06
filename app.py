from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from model import train_model, predict_failure

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sensor_data = request.json['sensor_data']
    prediction = predict_failure(sensor_data)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
