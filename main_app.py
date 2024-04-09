# main_app.py

from flask import Flask, render_template, request
from hd_prediction import PredictionPipeline
from d_prediction import PredictionPipeline_d
import numpy as np
import pickle
import numpy as np
from PIL import Image
import io

app = Flask(__name__)


# Load the trained model
model = pickle.load(open('image_classification_model.pkl', 'rb'))

def preprocess_image(image):
    # Convert the image to grayscale
    image = image.convert('L')
    # Resize the image to match the input size of the model
    image = image.resize((128, 128))
    # Convert image to numpy array
    image_array = np.asarray(image) / 255.0  # Normalize pixel values
    return image_array.reshape(-1, 128, 128, 1)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/heart_disease', methods=['GET', 'POST'])
def heart_disease_form():
    if request.method == 'POST':
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

        return render_template('heart_disease_form.html', predict=predicted_text)
    else:
        return render_template('heart_disease_form.html')



@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes_form():
    if request.method == 'POST':
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        bp = float(request.form['bp'])
        st = float(request.form['st'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = float(request.form['age'])
        

        data = [pregnancies, glucose, bp, st, insulin, bmi, dpf, age]
        data = np.array(data).reshape(1, 8)

        obj = PredictionPipeline_d()
        predict = obj.predict(data)

        if predict == 1:
            predicted_text = "Diabetes Predicted"
        else:
            predicted_text = "Diabetes Not Predicted"

        return render_template('diabetes_form.html', predict=predicted_text)
    else:
        return render_template('diabetes_form.html')



@app.route('/brain-tumor', methods=['GET', 'POST'])
def brain_tumor():
    if request.method == 'POST':
       # Get the uploaded image file from the request
        file = request.files['file']
        
        # Read the image file
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess the image
        input_image = preprocess_image(image)
        
        # Use the loaded model to predict the class label of the input image
        predicted_class = model.predict(input_image)
        
        # Convert the predicted class probabilities into a binary prediction
        threshold = 0.5  # Adjust the threshold as needed
        if predicted_class[0][1] < threshold:
            prediction = "Yes, the image has a brain tumor"
        else:
            prediction = "No, the image does not have a brain tumor"
        
        # Return the prediction
        return render_template('brain_tumor.html', prediction=prediction)
    else:
        return render_template('brain_tumor.html')

if __name__ == '__main__':
    app.run(debug=True)
