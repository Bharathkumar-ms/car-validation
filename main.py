from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Removed the following line since the static directory is not needed:
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Add CORS middleware with specific origins
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint to serve the main page
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    logger.info("Serving root endpoint")
    return templates.TemplateResponse("index.html", {"request": request})

# Constants
MODEL_PATH = os.getenv("MODEL_PATH", "mobilenet96.tflite")
IMG_HEIGHT, IMG_WIDTH = 224, 224
CLASS_LABELS = ['front', 'left', 'rear', 'right']

# Load the TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info(f"Successfully loaded TFLite model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load TFLite model: {str(e)}")
    raise RuntimeError(f"Failed to load TFLite model: {str(e)}")

# Preprocess image (async function)
async def preprocess_image(file: UploadFile):
    try:
        # Read the file content asynchronously
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        image = image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
        image = np.array(image, dtype=np.float32) / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

# Predict endpoint
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        logger.info("Received image for prediction")
        input_data = await preprocess_image(file)  # Use await since preprocess_image is async
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        predicted_idx = int(np.argmax(output_data))
        predicted_label = CLASS_LABELS[predicted_idx]
        confidence = float(output_data[predicted_idx])

        logger.info(f"Prediction: {predicted_label} with confidence {confidence}")
        return JSONResponse({
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "all_probabilities": {
                label: round(float(prob), 4)
                for label, prob in zip(CLASS_LABELS, output_data)
            }
        })
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    