from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')  # Save files inside 'static/uploads/'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max file size

# Load models only when needed
models = {
    'FCN': load_model('models/depression_detection_fcn.h5'),
    'FFL': load_model('models/depression_detection_ffl.h5'),
    'LSTM': load_model('models/depression_lstm_model.h5'),
    'HYBRID':load_model('models/depression_detection_hybrid_model.h5')
}

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'model_type' not in request.form:
        flash('No file or model type selected')
        return redirect(request.url)
    
    file = request.files['file']
    model_type = request.form['model_type']
    
    # Check if the file is valid and the model type is valid
    if file.filename == '' or model_type not in models:
        flash('Invalid file or model selection')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction using the selected model
        prediction = predict(filepath, model_type)
        
        # Pass model_name to display which model was used
        return render_template('result.html', filename=filename, prediction=prediction, model_name=model_type)
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)



def predict(filepath, model_type):
    img = Image.open(filepath)

    # Define correct input sizes for models
    model_input_shapes = {
        'FCN': (128, 128, 3),  
        'FFL': (128, 128, 3),  
        'LSTM': (48, 48, 1),
        'HYBRID':(128, 128, 3)
    }

    target_size, channels = model_input_shapes[model_type][:2], model_input_shapes[model_type][2]

    # Resize the image correctly based on the model
    img = img.resize(target_size)

    # Convert to grayscale if LSTM model (expects 1 channel)
    if model_type == 'LSTM':
        img = img.convert('L')  
    else:
        img = img.convert('RGB')  # Ensure 3 channels for FFL & FCN models

    # Convert image to numpy array
    img_array = np.array(img)

    # If grayscale, add a channel dimension
    if model_type == 'LSTM':
        img_array = np.expand_dims(img_array, axis=-1)  

    # Normalize image
    img_array = img_array / 255.0  

    # Expand dimensions to match model input (batch size 1)
    img_array = np.expand_dims(img_array, axis=0)  

    # Get the selected model
    model = models[model_type]

    # Make prediction
    prediction = model.predict(img_array)

    # Extract the first value from prediction
    predicted_value = float(prediction[0][0])  

    return 'Depressed 😢' if predicted_value < 0.5 else 'Not Depressed 😊'



if _name_ == '_main_':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
