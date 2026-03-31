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

    # Build input row with all zeros first
    input_dict = {col: 0 for col in model_columns}

    # Map fields from frontend to model columns
    input_dict['stars']                    = float(data.get('stars', 3.0))
    input_dict['review_count']             = float(data.get('review_count', 0))
    input_dict['avg_review_stars']         = float(data.get('avg_review_stars', 3.0))
    input_dict['review_count_total']       = float(data.get('review_count_total', 0))
    input_dict['days_since_last_review']   = float(data.get('days_since_last_review', 30))
    input_dict['business_age_days']        = float(data.get('business_age_days', 365))
    input_dict['rating_trend']             = float(data.get('rating_trend', 0))
    input_dict['latitude']                 = float(data.get('latitude', 36.17))
    input_dict['longitude']                = float(data.get('longitude', -115.14))
    input_dict['RestaurantsPriceRange2']   = float(data.get('RestaurantsPriceRange2', 2))

    # Amenities
    for field in [
        'RestaurantsGoodForGroups', 'RestaurantsDelivery', 'RestaurantsTakeOut',
        'RestaurantsReservations', 'WheelchairAccessible', 'BusinessAcceptsCreditCards',
        'WiFi', 'BikeParking', 'Caters', 'BusinessParking', 'CoatCheck',
        'ByAppointmentOnly', 'HasTV', 'OutdoorSeating', 'GoodForKids'
    ]:
        if field in input_dict:
            input_dict[field] = float(data.get(field, 0))

    input_df = pd.DataFrame([input_dict])[model_columns]

    prob = model.predict_proba(input_df)[0][1]
    # Model predicts closure risk, so success = 1 - closure_prob
    success_prob = 1 - prob
    score = round(success_prob * 100)

    if success_prob >= 0.75:
        decision, risk = "OPEN BUSINESS HERE", "LOW RISK"
    elif success_prob >= 0.50:
        decision, risk = "MODERATE RISK – INVESTIGATE FURTHER", "MEDIUM RISK"
    else:
        decision, risk = "HIGH FAILURE RISK – NOT RECOMMENDED", "HIGH RISK"

    return jsonify({
        "success_probability": round(float(success_prob), 4),
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
