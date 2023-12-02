from flask import Flask, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from flask import jsonify
import os
model = tf.keras.models.load_model('model.h5')


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'Empty file'})
            else:
                filename = secure_filename(file.filename)
                # sol1
                file.save('./static/images/'+filename)
                image = tf.keras.preprocessing.image.load_img(
                    ("./static/images/"+filename), target_size=(320, 320)
                )
                image = tf.keras.preprocessing.image.img_to_array(image)
                image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
                result = model.predict(tf.expand_dims(image, axis=0))[0]
                classes_x=np.argmax(result)
                return jsonify({'result': result.tolist(),
                                'class': str(classes_x)})
    else:
        return jsonify({'error': 'Method not allowed'})