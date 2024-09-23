from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('best_model.pkl')

@app.route('/')
def index():
    return "Heart Disease Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # You will get the data in JSON format
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
