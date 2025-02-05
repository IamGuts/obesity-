import joblib
import pandas as pd

class ObesityClassifier:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = joblib.load(f)

    def predict(self, input_data):
        return self.model.predict(input_data)