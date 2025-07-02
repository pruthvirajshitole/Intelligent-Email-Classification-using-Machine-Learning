import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer

import os

# Set up NLTK data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)

# Initialize tokenizer
try:
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_path)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



ps = PorterStemmer()

def transform_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Tokenization with error handling
        try:
            tokens = nltk.word_tokenize(text)
        except LookupError:
            # Fallback to basic splitting if NLTK tokenizer fails
            tokens = text.split()
        
        # Remove non-alphanumeric characters
        words = [word for word in tokens if word.isalnum()]
        
        # Remove stopwords and punctuation
        try:
            stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback to empty set if stopwords not available
            stop_words = set()
        
        words = [word for word in words if word not in stop_words and word not in string.punctuation]
        
        # Apply stemming
        stemmed_words = [ps.stem(word) for word in words]
        
        return " ".join(stemmed_words)
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return text  # Return original text if processing fails

# Load models with error handling
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models_loaded = False

st.title("Spam Email Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms:
        st.warning("Please enter a message to classify.")
    elif not models_loaded:
        st.error("Models are not properly loaded. Please check the application logs.")
    else:
        try:
            # 1. preprocess
            transformed_sms = transform_text(input_sms)
            
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            
            # 3. predict
            result = model.predict(vector_input)[0]
            
            # 4. Display
            if result == 1:
                st.header("Spam!!!")
            else:
                st.header("Not Spam!!!")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.info("Please try again with a different message.")
