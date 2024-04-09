import joblib 
import numpy as numpy
import pandas as pd 
from pathlib import Path


class PredictionPipeline_d:
    def __init__(self):
        self.model = joblib.load(Path("diabetes_model.pkl"))

    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction