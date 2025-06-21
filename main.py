from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and encoders
model = joblib.load('logistic_model.pkl')
le_device_type = joblib.load('le_device_type.pkl')
le_location = joblib.load('le_location.pkl')
le_manufacturer = joblib.load('le_manufacturer.pkl')
scaler_error = joblib.load('scaler_error.pkl')

current_date = datetime.now()

# Debug function to check encoder classes
def debug_encoders():
    print("=== ENCODER DEBUG INFO ===")
    print("Manufacturer classes:", le_manufacturer.classes_)
    print("Location classes:", le_location.classes_)
    print("Device type classes:", le_device_type.classes_)
    print("Model feature names (if available):", getattr(model, 'feature_names_in_', 'Not available'))
    print("Model coefficients shape:", model.coef_.shape if hasattr(model, 'coef_') else 'Not available')
    print("="*30)

# Call debug on startup
debug_encoders()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'Missing features'}), 400
    
    try:
        # Extract features from request
        manufacturer = data.get('manufacturer')
        location = data.get('location')
        deviceType = data.get('deviceType')
        eos_date_str = data.get('End_Of_Support_EOS_Date')
        eoss_date_str = data.get('End_Of_Security_Support_EOSS_Date')
        os_version_str = data.get('OSVersion')
        priority_str = data.get('priority')
        totalPhysicalMemoryProcessed = data.get('totalPhysicalMemoryProcessed')
        numberOfLogicalProcessorsProcessed = data.get('numberOfLogicalProcessorsProcessed')
        numberOfCoresProcessed = data.get('numberOfCoresProcessed')
        
        print(f"\n=== INPUT DEBUG ===")
        print(f"Manufacturer: {manufacturer}")
        print(f"Location: {location}")
        print(f"Device Type: {deviceType}")
        print(f"EOS Date: {eos_date_str}")
        print(f"EOSS Date: {eoss_date_str}")
        print(f"OS Version: {os_version_str}")
        print(f"Priority: {priority_str}")
        
        # Validate required fields
        required_fields = [manufacturer, location, deviceType, eos_date_str, eoss_date_str, 
                          os_version_str, priority_str, totalPhysicalMemoryProcessed,
                          numberOfLogicalProcessorsProcessed, numberOfCoresProcessed]
        
        if any(field is None for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Parse dates
        try:
            eos_date = pd.to_datetime(eos_date_str, format='%m/%d/%Y, %I:%M:%S %p')
            eoss_date = pd.to_datetime(eoss_date_str, format='%m/%d/%Y, %I:%M:%S %p')
        except:
            # Try alternative date formats
            try:
                eos_date = pd.to_datetime(eos_date_str)
                eoss_date = pd.to_datetime(eoss_date_str)
            except:
                return jsonify({'error': f'Invalid date format. Use MM/DD/YYYY, HH:MM:SS AM/PM'}), 400
        
        print(f"Parsed EOS Date: {eos_date}")
        print(f"Parsed EOSS Date: {eoss_date}")
        
        # Calculate device age (assuming device was purchased 5 years before EOS)
        device_purchase_date = eos_date - timedelta(days=5*365)
        deviceAgeYears = (current_date - device_purchase_date).days / 365
        print(f"Device Age (days): {deviceAgeYears}")
        
        # Extract OS version number - improved regex
        import re
        os_version_numbers = re.findall(r'\d+', os_version_str)
        if not os_version_numbers:
            return jsonify({'error': f'Could not extract OS version number from: {os_version_str}'}), 400
        OSVersion = float(os_version_numbers[0])  # Take first number found
        print(f"Extracted OS Version: {OSVersion}")
        
        # Map priority
        priority_map = {'PRIORITY_1': 1, 'PRIORITY_2': 2, 'PRIORITY_3': 3}
        priority = priority_map.get(priority_str)
        if priority is None:
            return jsonify({'error': f'Invalid priority value: {priority_str}. Use PRIORITY_1, PRIORITY_2, or PRIORITY_3'}), 400
        print(f"Priority: {priority}")
        
        # Calculate days until EOS and EOSS
        daysUntilEOS = (eos_date - current_date).days
        daysUntilEOSS = (eoss_date - current_date).days
        print(f"Days until EOS: {daysUntilEOS}")
        print(f"Days until EOSS: {daysUntilEOSS}")
        
        # Check if categorical values exist in encoders
        print(f"\n=== ENCODER CHECK ===")
        if manufacturer not in le_manufacturer.classes_:
            return jsonify({'error': f'Unknown manufacturer: {manufacturer}. Available: {list(le_manufacturer.classes_)}'}), 400
        if location not in le_location.classes_:
            return jsonify({'error': f'Unknown location: {location}. Available: {list(le_location.classes_)}'}), 400
        if deviceType not in le_device_type.classes_:
            return jsonify({'error': f'Unknown device type: {deviceType}. Available: {list(le_device_type.classes_)}'}), 400
        
        # Encode categorical variables
        manufacturer_encoded = le_manufacturer.transform([manufacturer])[0]
        location_encoded = le_location.transform([location])[0]
        deviceType_encoded = le_device_type.transform([deviceType])[0]
        
        print(f"Manufacturer '{manufacturer}' -> {manufacturer_encoded}")
        print(f"Location '{location}' -> {location_encoded}")
        print(f"Device Type '{deviceType}' -> {deviceType_encoded}")
        
        # Prepare feature array
        features = np.array([[
            manufacturer_encoded,
            location_encoded,
            deviceType_encoded,
            deviceAgeYears,
            OSVersion,
            priority,
            float(totalPhysicalMemoryProcessed),
            float(numberOfLogicalProcessorsProcessed),
            float(numberOfCoresProcessed),
            daysUntilEOS,
            daysUntilEOSS
        ]])
        features_encode = scaler_error.transform(features)
        print(f"\n=== FEATURE ARRAY ===")
        feature_names = [
            'manufacturer_encoded', 'location_encoded', 'deviceType_encoded',
            'deviceAgeYears', 'OSVersion', 'priority',
            'totalPhysicalMemoryProcessed', 'numberOfLogicalProcessorsProcessed',
            'numberOfCoresProcessed', 'daysUntilEOS', 'daysUntilEOSS'
        ]
        
        for i, (name, value) in enumerate(zip(feature_names, features[0])):
            print(f"{i}: {name} = {value}")
        
        print(f"Feature array shape: {features.shape}")
        print(f"Feature array: {features}")
        
        # Check for any extreme values
        print(f"\n=== VALUE CHECKS ===")
        print(f"Any NaN values: {np.isnan(features).any()}")
        print(f"Any infinite values: {np.isinf(features).any()}")
        print(f"Min value: {np.min(features)}")
        print(f"Max value: {np.max(features)}")
        
        # Make prediction
        prediction = model.predict(features_encode)
        prediction_proba = model.predict_proba(features_encode)
        
        print(f"\n=== PREDICTION RESULTS ===")
        print(f"Raw prediction: {prediction}")
        print(f"Raw probabilities: {prediction_proba}")
        print(f"Prediction shape: {prediction.shape}")
        print(f"Probabilities shape: {prediction_proba.shape}")
        
        return jsonify({
            'prediction': int(prediction[0]),
            'probability': {
                'class_0': float(prediction_proba[0][0]),
                'class_1': float(prediction_proba[0][1])
            },
            'debug_info': {
                'features': features.tolist()[0],
                'feature_names': feature_names,
                'input_validation': 'passed'
            }
        })
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"\n=== ERROR ===")
        print(error_traceback)
        return jsonify({
            'error': f'Prediction error: {str(e)}',
            'traceback': error_traceback
        }), 500

@app.route('/debug', methods=['GET'])
def debug_info():
    """Endpoint to get model and encoder information"""
    return jsonify({
        'model_type': str(type(model)),
        'manufacturer_classes': list(le_manufacturer.classes_),
        'location_classes': list(le_location.classes_),
        'device_type_classes': list(le_device_type.classes_),
        'model_features': getattr(model, 'feature_names_in_', 'Not available'),
        'model_coef_shape': model.coef_.shape if hasattr(model, 'coef_') else 'Not available'
    })

if __name__ == '__main__':
    app.run(debug=True)