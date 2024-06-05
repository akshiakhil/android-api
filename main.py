from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the Random Forest Classifier model
filename = 'random_forest_regression_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Weather Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user inputs from JSON request
            data = request.json
            t = int(data['T'])
            tm = int(data['TM'])
            tmm = int(data['Tm'])
            slp = int(data['SLP'])
            h = int(data['H'])
            vv = float(data['VV'])
            v = float(data['V'])
            vm = int(data['VM'])

            # Prepare input data for prediction
            input_data = np.array([[t, tm, tmm, slp, h, vv, v, vm]])
            my_prediction = classifier.predict(input_data)

            # Return the prediction as JSON response
            return jsonify({'prediction': my_prediction.tolist()})
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
