# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:45:03 2024

@author: Ndubisi M. Uzoegbu
"""
import pandas as pd
from flask import Flask, request, jsonify, render_template
import modules

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json
    user_input_df = pd.DataFrame(user_input)
    # Forecast price
    lin_pred = modules.forecast_price(user_input_df)
    numblen = len(lin_pred)
    # Create prediction results
    pr_results = {
        'Date': user_input_df.index[-numblen:].strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'lin_pred': (lin_pred + 0.0014).tolist()
    }
    
    return jsonify(pr_results)

if __name__ == '__main__':
    app.run(port=3010, debug=True)
