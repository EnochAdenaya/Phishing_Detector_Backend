from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = os.path.join(BASE_DIR, "random_forest_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        email_text = data.get("email")

        if not email_text:
            return jsonify({"error": "No email content provided"}), 400

        # Convert email text to feature vector
        email_vector = vectorizer.transform([email_text])

        # Predict
        prediction = model.predict(email_vector)[0]
        confidence = model.predict_proba(email_vector)[0][prediction]

        result = {
            "prediction": "Phishing" if prediction == 1 else "Legitimate",
            "confidence": round(confidence, 2)
        }

        return jsonify(result), 200  # HTTP 200 OK

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # HTTP 500 Internal Server Error


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)  # Allows access from all interfaces
