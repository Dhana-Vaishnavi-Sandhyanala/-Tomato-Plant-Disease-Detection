import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

MODEL_PATH = "model/tomato_model.h5"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path, class_names):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]

    return predicted_class

