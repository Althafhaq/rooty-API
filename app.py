
import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request 
from flask_cors import CORS, cross_origin

model = tf.keras.models.load_model('models/model.h5')


def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((150, 150))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img


def predict_result(img):
    predict_class = np.argmax(model.predict(img))
    if predict_class == 0:
        return 'Pisang Ambon'
    elif predict_class == 1:
        return 'Pisang Barangan'
    elif predict_class == 2:
        return 'Pisang Cavendish'
    elif predict_class == 3:
        return 'Pisang Kepok'
    elif predict_class == 4:
        return 'Pisang Nangka'
    elif predict_class == 5:
        return 'Pisang Susu'
    elif predict_class == 6:
        return 'Pisang Tanduk'

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/predict', methods=['POST'])
@cross_origin()
def infer_image():
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return jsonify(prediction=predict_result(img))
    

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return 'Rooty Fruity'


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')