from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import re

# pip install tensorflow==2.16.2 keras==3.6.0
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.utils import pad_sequences

from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import io
import base64

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
    # print(f"Preprocessed text: {text}")
    return text

def encode_lstm(text: str, max_length: int):
    try:
        def one_hot_encoded(text, vocab_size=5000):
            return one_hot(text, vocab_size)

        def word_embedding(text):
            tokens = text.split()
            preprocessed_text = " ".join(tokens)
            return one_hot_encoded(preprocessed_text)

        one_hot_encoded_input = word_embedding(text)
        padded_encoded_input = pad_sequences([one_hot_encoded_input], maxlen=max_length, padding='pre')
        return padded_encoded_input
    except Exception as e:
        print(f"Error during encoding: {e}")
        return np.array([])

def preprocess_query(query: str, max_length: int):
    processed_query = preprocessing(query)
    processed_query = encode_lstm(processed_query, max_length)
    return processed_query

def LimeExplanation(model, query, max_length):
    def predict_fn(texts):
        padded_inputs = [preprocess_query(text, max_length) for text in texts]
        padded_inputs = np.vstack(padded_inputs)
        predictions = model.predict(padded_inputs)
        if predictions.shape[1] == 1:
            return np.hstack([1 - predictions, predictions])
        return predictions

    # Run prediction and create LIME explanation
    probabilities = predict_fn([query])
    predicted_class = np.argmax(probabilities)
    explainer = LimeTextExplainer(class_names=['Fake', 'True'])
    explanation = explainer.explain_instance(query, predict_fn, num_features=10, top_labels=2)
    fig = explanation.as_pyplot_figure(label=predicted_class)

    # Extract influencing features
    positive_features = [
        feature for feature, weight in explanation.as_list(label=predicted_class)
        if weight > 0
    ]
    return fig, positive_features, predicted_class, probabilities[0]

# Define a route for prediction
@app.route("/api/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        content = data.get("content", "").lower()
        query = data.get("query", "")

        if content not in models_cache:
            raise ValueError(f"Model for content '{content}' is not preloaded.")

        # Retrieve the model and its max_length
        model_info = models_cache[content]
        model = model_info["model"]
        max_length = model_info["max_length"]

        # Preprocess and predict
        processed_query = preprocess_query(query, max_length)
        result = model.predict(processed_query)

        # Generate LIME explanation
        fig, positive_features, predicted_class, probabilities = LimeExplanation(model, query, max_length)

        # Convert LIME figure to Base64
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        img_buffer.close()

        # Return response
        return jsonify({
            "content": content,
            "query": query,
            "predicted_class": int(predicted_class),
            "probabilities": probabilities.tolist(),
            "indicators": positive_features,
            "lime_figure": img_base64,
        }), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500

# Main entry point
if __name__ == "__main__":
    preload_models()  # Preload models at startup
    app.run(debug=True, port=8080)
