from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define base directory for models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Directory containing all models

# Preprocessing function
def preprocess_query(query: str):
    processed = np.array([len(query)])  # Example: Replace with real preprocessing logic
    return processed

# Function to load the model dynamically based on 'content'
def load_dynamic_model(content: str):
    model_name = f"LSTM-without-tag-{content}.h5"  # Model naming convention
    model_path = os.path.join(MODELS_DIR, model_name)
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        raise FileNotFoundError(f"Model for content '{content}' not found at {model_path}")

# Define a route for prediction
@app.route("/api/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        content = data.get("content", "").lower()  # Content type: title, text, all
        query = data.get("query", "")

        # load the requested model
        model = load_dynamic_model(content)

        # Preprocess the query
        processed_query = preprocess_query(query)

        # Make prediction
        # prediction = model.predict(np.expand_dims(processed_query, axis=0))
        # result = bool(prediction[0][0] > 0.4)  # Binary classification
        result = model.predict(np.expand_dims(processed_query, axis=0))

        return jsonify({"query": query, "result": result, "model": content})
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    app.run(debug=True, port=8080)
