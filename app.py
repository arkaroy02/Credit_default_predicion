from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('lgb_model.pkl')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Extract features from the input JSON
    features = [data.get(col) for col in [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'PAY_2',
        'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
        'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
        'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
        'TOTAL_BILL', 'TOTAL_PAY', 'PAY_TO_BAL_RATIO'
    ]]
    
    # Ensure the features are in the correct shape
    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)[:, 1]

    # Prepare response
    response = {
        'prediction': int(prediction[0]),
        'prediction_proba': float(prediction_proba[0])
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
