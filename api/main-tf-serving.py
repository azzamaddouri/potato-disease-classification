from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer
import requests 

app = FastAPI()

# Load the model
endpoint = "http://localhost:8502/v1/models/potatoes_model:predict"

# Define class names (replace with your actual class names)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/")
async def root():
    return {"message": "Welcome to the Potato Disease API"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    json_data = {
        "instances": image_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    predictions = response.json()  # Assuming predictions is a dictionary
    prediction = np.array(predictions['predictions'][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "class": predicted_class,
        'confidence': confidence
    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
