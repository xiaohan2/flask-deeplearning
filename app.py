from flask import Flask, request, jsonify, make_response, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import base64
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
app = Flask(__name__)
CORS(app, supports_credentials=True)

img_basePath = './img_store/'
model_basePath = './model_store/'


@app.route('/')
def hello():
    return 'hello'


def my_predict(x, modelName):
    model = load_model(model_basePath + modelName)
    result = model.predict()
    return result


@app.route('/filerecv', methods=['POST'])
def receive():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(img_basePath + filename)
    with open('./' + filename, 'rb') as f:
        base64_data = base64.b64encode(f.read())
    return base64_data


@app.route('/getPredictedImage', method=['GET', 'POST'])
def getPredictedImage():
    modelName = request.args.get('model')
    imgName = request.args.get('img')
    img = image.load_img(img_basePath + imgName, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    result = my_predict(x, modelName)
