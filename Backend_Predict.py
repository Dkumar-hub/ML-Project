from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import keras.config

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ✅ Enable unsafe deserialization (needed for Lambda layers)
keras.config.enable_unsafe_deserialization()

# ✅ Define custom Lambda functions used in the model
def squeeze_last_dim(t):
    return K.squeeze(t, -1)

def expand_last_dim(t):
    return K.expand_dims(t, axis=-1)

custom_objects = {
    "squeeze_last_dim": squeeze_last_dim,
    "expand_last_dim": expand_last_dim
}

# ✅ Load the model
try:
    model = load_model("Predictions.h5", custom_objects=custom_objects)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ✅ Class labels
class_labels = ['Hello', 'Goodbye', 'Sorry', 'Thank You', 'No']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("🔵 /predict endpoint hit")
    try:
        data = request.json
        keypoints = data.get("keypoints", [])

        # Validate input shape
        if not keypoints or len(keypoints) != 30 * 126:
            return jsonify({"error": "Invalid input shape. Expected 30 frames with 126 keypoints each."}), 400

        input_array = np.array(keypoints).reshape(1, 30, 126)

        prediction = model.predict(input_array)[0]
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return jsonify({
            "prediction": predicted_class,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)