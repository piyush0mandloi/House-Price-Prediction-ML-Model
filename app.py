# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Model file path
MODEL_PATH = 'models/housing_model.pkl'
FEATURES_PATH = 'models/feature_names.pkl'

# Function to train and save model
def train_and_save_model():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load the California Housing dataset
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data['PRICE'] = housing.target
    
    # Split the data
    X = data.drop('PRICE', axis=1)
    y = data['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model and feature names
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(housing.feature_names, f)
    
    return model, housing.feature_names, data

# Load or train model
def get_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        # Load existing model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        with open(FEATURES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
            
        # Load data for min/max values
        housing = fetch_california_housing()
        data = pd.DataFrame(housing.data, columns=housing.feature_names)
        data['PRICE'] = housing.target
    else:
        # Train and save model
        model, feature_names, data = train_and_save_model()
    
    return model, feature_names, data

# Routes
@app.route('/')
def home():
    _, feature_names, data = get_model()
    
    # Get min, max, and default values for each feature
    feature_info = {}
    for feature in feature_names:
        feature_info[feature] = {
            'min': float(data[feature].min()),
            'max': float(data[feature].max()),
            'default': float(data[feature].median())
        }
    
    return render_template('index.html', feature_info=feature_info, features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    model, feature_names, _ = get_model()
    
    # Get values from form
    features = []
    for feature in feature_names:
        features.append(float(request.form.get(feature, 0)))
    
    # Make prediction
    input_features = np.array(features).reshape(1, -1)
    prediction = model.predict(input_features)[0]
    
    # Calculate feature contributions
    contributions = []
    for i, feature in enumerate(feature_names):
        contribution = float(features[i] * model.coef_[i])
        contributions.append({
            'feature': feature,
            'value': features[i],
            'coefficient': float(model.coef_[i]),
            'contribution': contribution
        })
    
    # Sort contributions by absolute value
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return jsonify({
        'prediction': float(prediction * 100000),  # Convert to actual dollars
        'contributions': contributions
    })

@app.route('/retrain', methods=['GET'])
def retrain():
    train_and_save_model()
    return jsonify({'status': 'success', 'message': 'Model retrained successfully'})

if __name__ == '__main__':
    app.run(debug=True)