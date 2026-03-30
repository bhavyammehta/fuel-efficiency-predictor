import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

class FuelEfficiencyModel:
    def __init__(self):
        self.model = None
        self.is_trained = False
        
        # ✅ FIX: Correct path
        self.model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "fuel_efficiency_model.joblib"
        )
        
    def generate_sample_data(self):
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'engine_size': np.random.uniform(1.0, 5.0, n_samples),
            'cylinders': np.random.randint(4, 8, n_samples),
            'horsepower': np.random.randint(80, 300, n_samples),
            'weight': np.random.randint(1500, 4000, n_samples),
            'acceleration': np.random.uniform(8, 20, n_samples),
            'model_year': np.random.randint(70, 85, n_samples),
            'origin': np.random.randint(1, 4, n_samples)
        }
        
        mpg = (45 - 2*data['engine_size'] - 0.5*data['cylinders'] + 
               0.1*data['horsepower'] - 0.005*data['weight'] + 
               0.5*data['acceleration'] + 0.3*(data['model_year'] - 70) +
               np.random.normal(0, 2, n_samples))
        
        data['mpg'] = np.maximum(mpg, 10)
        
        return pd.DataFrame(data)
    
    def train_model(self):
        df = self.generate_sample_data()
        
        features = ['engine_size', 'cylinders', 'horsepower', 'weight', 
                    'acceleration', 'model_year', 'origin']
        
        X = df[features]
        y = df['mpg']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.is_trained = True
        
        # ✅ FIX: Save using absolute path
        joblib.dump(self.model, self.model_path)
        
        return mae, r2
    
    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                return True
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, features):
        if not self.is_trained:
            if not self.load_model():
                self.train_model()
        
        feature_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(feature_array)[0]
        
        return round(prediction, 2)