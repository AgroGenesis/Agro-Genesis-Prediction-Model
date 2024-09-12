from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np  # Make sure numpy is imported

app = Flask(__name__)

# Load the pre-trained models and preprocessing tools
# Recommendation Models
best_forest = joblib.load('models/crop_recommendation_model.pkl')
scaler = joblib.load('models/crop_recommendation_scaler.pkl')
pca = joblib.load('models/crop_recommendation_pca.pkl')
encoder = joblib.load('models/crop_recommendation_label_encoder.pkl')
# Crop Yield Prediction Model
yield_prediction_model = joblib.load('models/yield_prediction_model.pkl')

# Function to predict the most suitable crop using the loaded model
def predict_crop(n, p, k, temperature, humidity, ph, rainfall):
    try:
        print("Recommendation Model")
        # Create a DataFrame with the same columns used during training
        input_features = pd.DataFrame(
            [[n, p, k, temperature, humidity, ph, rainfall]],
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        )
        
        # Scale the input features
        input_features_scaled = scaler.transform(input_features)
        
        # Apply PCA transformation (Make sure PCA was trained with enough components)
        input_features_pca = pca.transform(input_features_scaled)
        
        # Predict using the loaded RandomForest model
        rf_pred = best_forest.predict(input_features_pca)
        rf_pred_crop = encoder.inverse_transform(rf_pred)
        
        return rf_pred_crop[0]
    except Exception as e:
        return str(e)

# API endpoint for crop recommendation
@app.route('/recommendation', methods=['POST'])
def recommended():
    try:
        # Parse JSON request body
        data = request.get_json()
        
        # Extract parameters from request
        n = data['n']
        p = data['p']
        k = data['k']
        temperature = data['temperature']
        humidity = data['humidity']
        ph = data['ph']
        rainfall = data['rainfall']
        
        # Call the prediction function
        recommended_crop = predict_crop(n, p, k, temperature, humidity, ph, rainfall)
        
        print(recommended_crop)
        return jsonify({
            'recommended_crop': recommended_crop
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def yield_prediction(area, season, crop):
    try:
        # Create a DataFrame with the input data
        input_features = pd.DataFrame(
            [[area, season, crop]],
            columns=['Area', 'Season', 'Crop']
        )
        
        # Create dummy variables for categorical columns
        input_dummy = pd.get_dummies(input_features)
        
        # Align the input dummy variables with the training dataset (fill missing columns with 0)
        input_dummy = input_dummy.reindex(columns=yield_prediction_model.feature_names_in_, fill_value=0)
        
        # Scale the input features using the same scaler used in training
        input_scaled = scaler.transform(input_dummy)
        
        # Apply PCA transformation if applicable
        if pca:
            input_pca = pca.transform(input_scaled)
        else:
            input_pca = input_scaled
        
        # Predict using the loaded RandomForest model
        rf_pred = best_forest.predict(input_pca)
        rf_pred = np.maximum(0, rf_pred)  # Ensure no negative values
        rf_pred = np.round(rf_pred, 2)  # Round for readability

        # Predict using the other model (if needed)
        crop_pred = yield_prediction_model.predict(input_pca)
        crop_pred = np.maximum(0, crop_pred)  # Ensure no negative values
        crop_pred = np.round(crop_pred, 2)  # Round for readability

        return rf_pred[0], crop_pred[0]
    except Exception as e:
        return str(e), None

# API endpoint for crop prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request body
        print("Crop Yield Prediction Model")
        data = request.get_json()
        
        # Extract parameters from request
        area = data['Area']
        season = data['Season']
        crop = data['Crop']
        
        # Call the prediction function
        rf_prediction, crop_prediction = yield_prediction(area, season, crop)
        print(rf_prediction, crop_prediction)
        return jsonify({
            'rf_prediction': rf_prediction,
            'crop_prediction': crop_prediction
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)



