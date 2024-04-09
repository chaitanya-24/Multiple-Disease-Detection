from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from hd_prediction import PredictionPipeline

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            cp = float(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = float(request.form['fbs'])
            restecg = float(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = float(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = float(request.form['slope'])
            ca = float(request.form['ca'])
            thal = float(request.form['thal'])

            data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            data = np.array(data).reshape(1, 13)

            obj = PredictionPipeline()
            predict = obj.predict(data)

            if predict == 1:
                predicted_text = "Heart Disease Predicted"
            else:
                predicted_text = "Heart Disease Not Predicted"

            return render_template('index1.html', predict=predicted_text)

        except Exception as e:
            return 'Something is wrong: ${e}'
    else:
        return render_template('index1.html')
    

if __name__ == "__main__":
    app.run(debug=True)
