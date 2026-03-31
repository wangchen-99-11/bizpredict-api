from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    prob = model.predict_proba(input_df)[0][1]
    score = round(prob * 100)
    if prob >= 0.75:
        decision, risk = "OPEN BUSINESS HERE", "LOW RISK"
    elif prob >= 0.50:
        decision, risk = "MODERATE RISK – INVESTIGATE FURTHER", "MEDIUM RISK"
    else:
        decision, risk = "HIGH FAILURE RISK – NOT RECOMMENDED", "HIGH RISK"
    return jsonify({
        "success_probability": round(float(prob), 4),
        "success_score": score,
        "decision": decision,
        "risk_level": risk
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
