import warnings
warnings.filterwarnings('ignore')
import joblib
import numpy as np
import cv2
import tensorflow

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model and label binarizer
model = load_model('Skin_disease.h5')
lb = joblib.load('label_binarizer.pkl')

def find_disease(image_file):

    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid image path"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))

    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    label = lb.inverse_transform(prediction)

    return label[0]

# Example
