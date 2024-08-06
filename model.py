import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Train a simple model (for demonstration purposes)
def train_model():
    # Example training data
    data = pd.DataFrame({
        'sensor_1': np.random.rand(100),
        'sensor_2': np.random.rand(100),
        'failure': np.random.randint(0, 2, 100)
    })
    
    X = data[['sensor_1', 'sensor_2']]
    y = data['failure']
    
    model = RandomForestClassifier()
    model.fit(X, y)
    
    joblib.dump(model, 'model.pkl')

def predict_failure(sensor_data):
    model = joblib.load('model.pkl')
    data = np.array(sensor_data).reshape(1, -1)
    prediction = model.predict(data)
    return prediction[0]

# Train the model when script is run
if __name__ == '__main__':
    train_model()
