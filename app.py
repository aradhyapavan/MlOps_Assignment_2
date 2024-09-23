from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your pre-trained model
model = joblib.load('best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data
        data = request.get_json(force=True)
        
        # Validate input data
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key in request'}), 400
        
        features = data['features']
        
        if not isinstance(features, list) or len(features) != 18:
            return jsonify({'error': 'Features should be a list of 18 numerical values'}), 400
        
        # Convert to numpy array
        input_data = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Return prediction as a standard Python int
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)
