from PIL import Image
import numpy as np
import pickle
import io
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Model class to define the architecture
class Model:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)  # Example model
        self.scaler = StandardScaler()

    def load(self, model_path, scaler_path):
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Error loading file: {e}")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def predict(self, X):
        return self.model.predict(X)

def get_remedy(plant_disease):
    try:
        with open("model_files/data.json", 'r') as f:
            remedies = json.load(f)
        return remedies.get(plant_disease, "Not Found!")
    except Exception as e:
        print(f"Error loading remedies: {e}")
        return "Error"

def predict_plant(model, imgdata):
    try:
        # Load labels from a JSON file
        with open('model_files/labels.json', 'rb') as lb:
            labels = pickle.load(lb)

        # Convert Base64 string to Image
        image = Image.open(io.BytesIO(imgdata)).convert('RGB')  # Ensure RGB format
        # Resize and normalize the Image
        image = image.resize((256, 256))
        image = np.array(image) / 255.0  # Normalize image to [0, 1]
        image_flattened = image.flatten().reshape(1, -1)  # Flatten image for model input

        # Use the scaler to normalize the input data
        image_scaled = model.scaler.transform(image_flattened)

        # Getting prediction from model
        y_result = model.predict(image_scaled)
        result_idx = y_result[0]

        # Get plant disease from result
        plant_disease = next((key for key, value in labels.items() if value == result_idx), "Unknown")

        if "healthy" not in plant_disease:
            # Get remedy for given plant disease
            remedy = get_remedy(plant_disease)
        else:
            remedy = "Plant is Healthy"

        return plant_disease, remedy
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", "Error"

# Example usage
if __name__ == "__main__":
    model = Model()
    model.load("model_files/model.pkl", "model_files/scaler.pkl")  # Load your model and scaler

    # Example: predicting with a given image data
    try:
        with open("src\model_files\test\PotatoEarlyBlight1.JPG", "rb") as img_file:
            imgdata = img_file.read()
            disease, remedy = predict_plant(model, imgdata)
            print(f"Disease: {disease}, Remedy: {remedy}")
    except Exception as e:
        print(f"Error reading image file: {e}")
