import joblib 
import numpy as numpy
import pandas as pd 
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path("rf_model_hd.pkl"))

    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction