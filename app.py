# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:45:03 2024

@author: Ndubisi M. Uzoegbu
"""

from flask import Flask, request, jsonify
import modules

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json
    user_input_df = pd.DataFrame(user_input)
    # Forecast price
    lin_pred, rf_pred = forecast_price(user_input_df)
    numblen = len(lin_pred)
    # Create prediction results
    pr_results = {
        'Date': user_input_df.index[-numblen:].strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'lin_pred': (lin_pred + 0.0014).tolist(),
        'rf_pred': (rf_pred + 0.0014).tolist()
    }
    
    return jsonify(pr_results)

if __name__ == '__main__':
    app.run(debug=True)
