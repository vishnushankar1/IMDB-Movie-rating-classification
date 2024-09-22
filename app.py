import numpy as np
import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the word index for IMDb
word_index = imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

# Load the model

import streamlit as st
import logging
from keras.models import load_model

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
try:
    model = load_model('simple_rnn_imdb.h5')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Error loading model: %s", e)
    st.error("Failed to load the model. Please check the logs.")

# Your app code follows here...


# Helper Functions
def decode_review(text):
    return ' '.join(reverse_word_index.get(i-3, '?') for i in text)

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(i, 2)+3 for i in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocessed_text = preprocess_text(review)
    prediction = model.predict(preprocessed_text)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write("Enter a movie review to classify it as **positive** or **negative**.")

user_input = st.text_area("Movie Review", placeholder="Type your review here...", help="Write the review you want to classify.")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please provide a review before clicking the classify button!")
    else:
        preprocessed_text = preprocess_text(user_input)
        prediction = model.predict(preprocessed_text)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        
        st.write(f"The Review is **{sentiment}**.")
        st.write(f"The confidence of the prediction is **{prediction[0][0]:.2f}**.")
else:
    st.write("Please click the button to classify the review.")
