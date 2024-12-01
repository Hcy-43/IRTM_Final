from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import pad_sequences

# pip install tensorflow==2.16.2 keras==3.6.0

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define base directory for models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Directory containing all models

# Dictionary to store preloaded models
models_cache = {}
def preload_models():
    print("Preloading models...")
    model_files = {
        "title": {"file": "LSTM-without-tag-title.h5", "max_length": 42},
        "text": {"file": "LSTM-without-tag-text.h5", "max_length": 8134},
        "all": {"file": "LSTM-without-tag-all.h5", "max_length": 8147},
    }
    for model_name, model_info in model_files.items():
        model_path = os.path.join(MODELS_DIR, model_info["file"])
        if os.path.exists(model_path):
            models_cache[model_name] = {
                "model": load_model(model_path),
                "max_length": model_info["max_length"],
            }
            # print(f"Loaded model: {model_name} with max_length: {model_info['max_length']} from {model_path}")
        else:
            print(f"Model file not found: {model_path}")
    print("Model preloading complete.")

# Preprocess text
def preprocessing(text):
    try:
        text = str(text).encode("utf-8", "ignore").decode("utf-8", "ignore").lower()
    except Exception as e:
        print(f"Error decoding text: {e}")
        text = ""
    text = text.replace('(reuters)', '').replace('rueters', '').replace('reuters', '').replace('reuter’s', '').replace('reuter', '')
    text = text.replace('\n', '')
    text = re.sub(r'http\S+|www\.\S+', '', text)
    print(f"Preprocessed text: {text}")
    return text

def encode_lstm(text: str, max_length=40):
    try:
        def one_hot_encoded(text, vocab_size=5000):
            return one_hot(text, vocab_size)

        def word_embedding(text):
            tokens = text.split()
            preprocessed_text = " ".join(tokens)
            return one_hot_encoded(preprocessed_text)

        one_hot_encoded_input = word_embedding(text)
        padded_encoded_input = pad_sequences([one_hot_encoded_input], maxlen=max_length, padding='pre')
        # print(f"Padded input: {padded_encoded_input}")
        return padded_encoded_input
    except Exception as e:
        print(f"Error during encoding: {e}")
        return np.array([])
    

def preprocess_query(query: str, max_length: int):
    print(f"Processing query: {query}")
    try:
        query = query.encode("utf-8", "ignore").decode("utf-8", "ignore")
        # print(f"Cleaned query: {query}")
    except Exception as e:
        print(f"Error cleaning query: {e}")
        query = ""
    processed_query = preprocessing(query)
    # print(f"Preprocessed query: {processed_query}")
    processed_query = encode_lstm(processed_query, max_length)
    # print(f"Encoded query: {processed_query}")
    return processed_query


# Define a route for prediction
@app.route("/api/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        content = data.get("content", "").lower()  # Model identifier: title, text, or all
        query = data.get("query", "")

        if content not in models_cache:
            raise ValueError(f"Model for content '{content}' is not preloaded.")

        # Retrieve the model and its max_length
        model_info = models_cache[content]
        model = model_info["model"]
        max_length = model_info["max_length"]

        # Preprocess the query with the correct max_length
        processed_query = preprocess_query(query, max_length=max_length)

        # Debugging shapes
        # print(f"Processed query shape: {processed_query.shape}")
        # print(f"Model expected input shape: {model.input_shape}")

        # Predict using the model
        result = model.predict(processed_query)
        return jsonify({"query": query, "result": bool(result.tolist()[0][0] > 0.4), "prob":  result.tolist()[0][0], "model": content})
        # return jsonify({"query": query, "prob":  result.tolist()[0][0], "model": content}) 
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500

# Main entry point
if __name__ == "__main__":
    preload_models()  # Preload models at startup
    app.run(debug=True, port=8080)
