from flask import Flask, request, render_template, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import os

# Load your pre-trained model for chicken classification
chicken_model = tf.keras.models.load_model('D:/Jupyter/healthy_defect.keras')

# Load your pre-trained model for defect classification
defect_model = tf.keras.models.load_model('D:/Jupyter/broken_model.keras')

# Initialize Flask application
app = Flask(__name__)

# Define a function to process the uploaded image
def process_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize image to match MobileNetV2 input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image
    return img, img_array

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for image upload and classification
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # Check if the file has a name
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # Process the uploaded image
        img, img_array = process_image(file)

        # Make predictions for chicken classification
        chicken_predictions = chicken_model.predict(img_array)

        # Determine the class label for chicken classification
        if chicken_predictions[0][0] > chicken_predictions[0][1]:
            chicken_label = "Defect Chicken"
        else:
            chicken_label = "Healthy Chicken"

        # If the chicken is classified as Defect, proceed with defect classification
        if chicken_label == "Defect Chicken":
            defect_predictions = defect_model.predict(img_array)
            if defect_predictions[0][0] > defect_predictions[0][1]:
                defect_label = "Broken"
            else:
                defect_label = "Hematoma"
            message = f"Defect Chicken - {defect_label}"
        else:
            message = chicken_label

        # Save the uploaded image temporarily
        image_path = os.path.join('static', file.filename)
        img.save(image_path)

        # Pass the image path and prediction to the template
        return render_template('index.html', message=message, image=image_path)

if __name__ == '__main__':
    app.run(debug=True)
