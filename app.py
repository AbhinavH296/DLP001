import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary nltk resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

# Load the trained model and tokenizer
model = load_model('best_sentiment_model.h5')
tokenizer = load('tokenizer.joblib')

# Define maximum sequence length (same as during training)
max_len = 100

# Function to preprocess and predict sentiment
def preprocess_and_predict(text):
    """Preprocess the input text and predict the sentiment."""
    # Clean the text (same as during training)
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces
    text = text.strip().lower()  # Convert to lowercase
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in STOPWORDS]  # Lemmatize and remove stopwords
    cleaned_text = " ".join(words)

    # Tokenize the cleaned text
    text_sequence = tokenizer.texts_to_sequences([cleaned_text])
    text_padded = pad_sequences(text_sequence, maxlen=max_len, padding='post', truncating='post')

    # Predict sentiment
    pred = model.predict(text_padded)
    sentiment = np.argmax(pred, axis=1)[0]
    
    # Calculate confidence scores for each sentiment class
    sentiment_scores = pred[0]
    
    # Convert the prediction to sentiment labels
    label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    sentiment_label = label_mapping[sentiment]
    
    return sentiment_label, sentiment_scores

# Streamlit app layout
st.set_page_config(page_title="Sentiment Analysis Web App", page_icon=":bar_chart:", layout="wide")
st.title("Sentiment Analysis Web App")
st.markdown(
    """
    <style>
    .header {
        text-align: center;
        font-size: 36px;
        color: #333;
    }
    .description {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header">Analyze the Sentiment of Your Text!</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Enter a sentence or tweet to get the sentiment: Negative, Neutral, or Positive.</div>', unsafe_allow_html=True)

# User input
user_input = st.text_area("Text Input", "", height=150)

if st.button("Predict Sentiment"):
    if user_input:
        sentiment, sentiment_scores = preprocess_and_predict(user_input)
        
        # Display sentiment label
        st.subheader(f"Sentiment: **{sentiment}**")
        
        # Visualize the sentiment prediction confidence scores using a bar chart
        labels = ['Negative', 'Neutral', 'Positive']
        confidence_df = pd.DataFrame({
            'Sentiment': labels,
            'Confidence': sentiment_scores
        })
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Sentiment', y='Confidence', data=confidence_df, palette="coolwarm", ax=ax)
        ax.set_title("Sentiment Confidence Scores")
        ax.set_ylabel('Confidence')
        st.pyplot(fig)
        
        # Show the confidence scores numerically
        st.write(f"Confidence Scores: \nNegative: {sentiment_scores[0]*100:.2f}%\nNeutral: {sentiment_scores[1]*100:.2f}%\nPositive: {sentiment_scores[2]*100:.2f}%")
    else:
        st.warning("Please enter some text to analyze.")
