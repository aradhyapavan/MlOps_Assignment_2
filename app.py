from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load your pre-trained model once when the app starts
try:
    model = joblib.load('best_model.pkl')
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error("Model file not found. Ensure 'best_model.pkl' exists.")
    model = None

# List of features with short descriptions and possible values
feature_descriptions = {
    'age': 'Age of the patient (integer, e.g., 45, 63).',
    'sex': 'Gender (1 = Male, 0 = Female).',
    'cp': 'Chest pain type (1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal, 4 = Asymptomatic).',
    'trestbps': 'Resting blood pressure (in mm Hg, e.g., 120, 145).',
    'chol': 'Serum cholesterol in mg/dl (e.g., 200, 233).',
    'fbs': 'Fasting blood sugar (1 = True if > 120 mg/dl, else 0).',
    'restecg': 'Resting ECG results (0 = Normal, 1 = ST-T abnormality, 2 = LVH).',
    'thalach': 'Max heart rate achieved (e.g., 150, 170).',
    'exang': 'Exercise-induced angina (1 = Yes, 0 = No).',
    'oldpeak': 'ST depression induced by exercise (float, e.g., 1.0, 2.3).',
    'slope': 'Slope of the peak exercise ST segment (0 = Upsloping, 1 = Flat, 2 = Downsloping).',
    'ca': 'Number of major vessels colored by fluoroscopy (0-3).',
    'thal': 'Thalassemia (3 = Normal, 6 = Fixed defect, 7 = Reversible defect).',
    'cp_2': 'One-hot for chest pain type 2 (1 = Yes, 0 = No).',
    'cp_3': 'One-hot for chest pain type 3 (1 = Yes, 0 = No).',
    'cp_4': 'One-hot for chest pain type 4 (1 = Yes, 0 = No).',
    'restecg_1': 'One-hot for resting ECG ST-T abnormality (1 = Yes, 0 = No).',
    'restecg_2': 'One-hot for resting ECG LVH (1 = Yes, 0 = No).',
    'slope_2': 'One-hot for downsloping ST segment (1 = Yes, 0 = No).'
}

@app.route('/', methods=['GET'])
def home():
    feature_info = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Heart Disease Prediction API</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
        <!-- Bootstrap Icons CDN -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.8.1/font/bootstrap-icons.min.css">
    </head>
    <body>
        <div class="container mt-5 text-center">
            <img src="https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/http://coursera-university-assets.s3.amazonaws.com/b9/c608c79b5c498a8fa55b117fc3282f/5.-Square-logo-for-landing-page---Alpha.png?auto=format%2Ccompress&dpr=1&w=180&h=180" alt="BITS Pilani Logo" class="mb-4">
            <h1 class="text-center mb-4">Welcome to the Heart Disease Prediction API</h1>
            <p class="lead text-center">Use the API to predict the likelihood of heart disease based on the features provided below.</p>
            
            <!-- MLOps Group Details -->
            <div class="text-start my-5 p-4 card">
            
                <h2>MLOps (S2-23_AIMLCZG523) - Assignment 2</h2>
                <h3>Group Number - 56</h3>
                <h4>Group Members:</h4>
                <p>
                    AKASH S (2022ac05316)<br>
                    APURBA ROY (2022ac05075)<br>
                    ARADHYA PAVAN H S (2022ac05457)<br>
                    GAYATHRY S (2022ac05263)<br>
                    JESINTA ROZARIO (2022ac05566)
                </p>
            </div>

            <!-- Available Features -->
            <div class="card mb-5 text-start">
                <div class="card-header bg-primary text-white">
                    <h2>Available Features</h2>
                </div>
                <div class="card-body">
                    <ul class="list-group">
    '''
    # Add each feature with its description
    for feature, description in feature_descriptions.items():
        feature_info += f'<li class="list-group-item"><strong>{feature}:</strong> {description}</li>'

    feature_info += '''
                    </ul>
                </div>
            </div>

            <!-- cURL Command Section -->
            <div class="d-flex justify-content-between align-items-center mb-3">
                <h3>Example Request Body:</h3>
                <button class="btn btn-primary mb-3" onclick="copyToClipboard()">
                    <i class="bi bi-clipboard"></i> <!-- Clipboard icon -->
                </button>
            </div>
            
            <!-- Copied message display -->
            <p id="copyMessage" class="text-muted mb-3"></p>

            <pre id="curlCommand" class="bg-light p-3 text-start">curl -X POST http://localhost:5000/predict \\
-H "Content-Type: application/json" \\
-d '{
  "features": [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 1, 0, 1, 0, 1, 0, 0, 1]
}'</pre>
        </div>


        <script>
            function copyToClipboard() {
                var curlText = document.getElementById('curlCommand').innerText;
                navigator.clipboard.writeText(curlText).then(function() {
                    var messageElement = document.getElementById('copyMessage');
                    messageElement.innerText = "Copied to clipboard!";
                    setTimeout(function() {
                        messageElement.innerText = ""; // Clears the message after 5 seconds
                    }, 3000); // 3000 milliseconds = 3 seconds
                }, function(err) {
                    document.getElementById('copyMessage').innerText = "Failed to copy: " + err;
                });
            }
        </script>

        <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.8/dist/umd/popper.min.js" integrity="sha384-I7E8VVD/ismYTF4hNIPjVp/Zjvgyol6VFvRkX/vR+Vc4jQkC+hVqc2pM8ODewa9r" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.min.js" integrity="sha384-0pUGZvbkm6XF6gxjEnlmuGrJXVbNuzT9qBBavbLwCsOGabYfZo0T0to5eqruptLy" crossorigin="anonymous"></script>
    </body>
    </html>
    '''
    return feature_info

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the server logs.'}), 500

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
