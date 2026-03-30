from flask import Flask, render_template, request, jsonify
import sys
import os
# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import FuelEfficiencyModel
    print("Successfully imported FuelEfficiencyModel")
except ImportError as e:
    print(f"Import error: {e}")
    print("Current directory:", os.path.dirname(os.path.abspath(__file__)))
    print("Files in directory:", os.listdir('.'))
    raise

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Initialize model
print("Initializing model...")
model = FuelEfficiencyModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        engine_size = float(request.form['engine_size'])
        cylinders = int(request.form['cylinders'])
        horsepower = int(request.form['horsepower'])
        weight = int(request.form['weight'])
        acceleration = float(request.form['acceleration'])
        model_year = int(request.form['model_year'])
        origin = int(request.form['origin'])
        
        # Prepare features for prediction
        features = [engine_size, cylinders, horsepower, weight, 
                   acceleration, model_year, origin]
        
        # Make prediction
        prediction = model.predict(features)
        
        # Determine efficiency category
        if prediction >= 30:
            category = "Excellent"
            color = "success"
        elif prediction >= 20:
            category = "Good"
            color = "info"
        elif prediction >= 15:
            category = "Average"
            color = "warning"
        else:
            category = "Poor"
            color = "danger"
        
        return render_template('result.html', 
                             prediction=prediction,
                             category=category,
                             color=color,
                             features=request.form)
    
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        print(error_msg)
        return render_template('index.html', error=error_msg)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        features = [
            float(data['engine_size']),
            int(data['cylinders']),
            int(data['horsepower']),
            int(data['weight']),
            float(data['acceleration']),
            int(data['model_year']),
            int(data['origin'])
        ]
        
        prediction = model.predict(features)
        
        return jsonify({
            'predicted_mpg': prediction,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/train', methods=['GET'])
def train_model():
    try:
        mae, r2 = model.train_model()
        return jsonify({
            'message': 'Model trained successfully!',
            'mean_absolute_error': round(mae, 2),
            'r2_score': round(r2, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Ensure model is loaded or trained
    print("Starting Flask application...")
    if not model.load_model():
        print("Training model on startup...")
        model.train_model()
    else:
        print("Model loaded successfully!")
    
    print("Flask app starting...")
    app.run(debug=True, host='127.0.0.1', port=5000)