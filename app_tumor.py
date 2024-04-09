from flask import Flask, render_template, request
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


@app.route('/', methods=['GET'])
def index():
    return render_template('index_t.html')

@app.route('/predict', methods=['POST'])
def predict():
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
    return render_template('index_t.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
