from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [
        data['Heart Rate'],
        data['Respiratory Rate'],
        data['Body Temperature'],
        data['Oxygen Saturation'],
        data['Systolic Blood Pressure'],
        data['Diastolic Blood Pressure'],
        data['Age'],
        1 if data['Gender'].lower() == 'male' else 0,
        data['Weight (kg)'],
        data['Height (m)'],
        data['Derived_HRV'],
        data['Derived_Pulse_Pressure'],
        data['Derived_MAP']
    ]
    input_array = np.array([features])
    prediction = model.predict(input_array)[0]
    return jsonify({'prediction': str(prediction)})
if __name__ == '__main__':
    app.run(debug=True)
