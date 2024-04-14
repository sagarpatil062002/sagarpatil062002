from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return "Hello Aniket"

@app.route('/predict', methods=['GET'])
def predict():
    # Get input data from the request form
    N = float(request.form.get('N'))
    P = float(request.form.get('P'))
    K = float(request.form.get('K'))
    temperature = float(request.form.get('temperature'))
    humidity = float(request.form.get('humidity'))
    ph = float(request.form.get('ph'))
    rainfall = float(request.form.get('rainfall'))

    # Convert input data to numpy array
    input_query = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Make prediction
    result = model.predict(input_query)

    # Return the result as JSON
    return jsonify({'crop': result[0]})

if __name__ == '__main__':
    app.run(debug=True)
