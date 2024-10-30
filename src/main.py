from flask import Flask, jsonify, request
from flask_cors import CORS
from model_files.ml_predict import predict_plant  # Ensure Network is defined or removed
import base64

app = Flask("Plant Disease Detector")
CORS(app)

@app.route('/', methods=['POST'])
def predict():
    try:
        key_dict = request.get_json()
        image = key_dict.get("image")

        if not image:
            return jsonify({"error": "Image data is required"}), 400

        imgdata = base64.b64decode(image)

        # Initialize your model here (ensure Network is correctly defined)
        model = Network()  # Make sure Network is defined in ml_predict.py

        # Call predict_plant to get results
        result, remedy = predict_plant(model, imgdata)

        # Process result to extract plant and disease
        plant, disease = result.split("___") if "___" in result else (result, "Unknown")
        disease = " ".join(disease.split("_"))  # Format the disease name

        # Prepare the response
        response = {
            "plant": plant,
            "disease": disease,
            "remedy": remedy,
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
