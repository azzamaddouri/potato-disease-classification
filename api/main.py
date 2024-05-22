from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# Load the model
MODEL_PATH = "../saved_models/1"
MODEL = TFSMLayer(MODEL_PATH, call_endpoint='serving_default')

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
    
    # Perform prediction
    predictions = MODEL(image_batch)
    print("Predictions:", predictions)  # Log predictions object

    # Process predictions
    predicted_class_index = np.argmax(predictions['output_0'][0])
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions['output_0'][0]))

    return {
        "class": predicted_class,
        'confidence' : confidence
    }



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
