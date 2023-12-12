import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import io

from util import classify, set_background

set_background('./BG/bg.jpg')

# set title
st.title('Pneumonia COVID Classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('./model/PMAFP.h5')

# Explicitly compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# load class names
with open('./model/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

# display image and perform classification
if file is not None:
    try:
        # Open the image using BytesIO
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        st.image(image, use_column_width=True)

        # classify image
        class_name, conf_score = classify(image, model, class_names)

        # write classification
        st.write("## {}".format(class_name))
        st.write("### Score: {}%".format(int(conf_score * 1000) / 10))

    except Exception as e:
        st.error(f"Error processing image: {e}")
