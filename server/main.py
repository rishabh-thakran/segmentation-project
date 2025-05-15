import sys
import os
import io
import base64
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
from torchvision import transforms
import logging
from fastapi.middleware.cors import CORSMiddleware

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import different segmentation models
from models.unet import UNet
from models.deeplabv3 import DeepLabV3
from models.pspnet import PSPNet

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to ["http://localhost:3000"] for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for saved predictions
SAVE_DIR = "../training/saved_predictions"

# Model dictionary
models = {
    "unet": UNet(num_classes=150),
    "deeplabv3": DeepLabV3(num_classes=150),
    "pspnet": PSPNet(num_classes=150),
}

# Load trained weights for models
for model_name, model in models.items():
    model.load_state_dict(torch.load(f"../training/saved_models/{model_name}.pth"))
    model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_prediction(model_name, epoch):
    pred_path = f"{SAVE_DIR}/{model_name}_epoch_{epoch}.npy"
    if not os.path.exists(pred_path):
        return None
    pred = np.load(pred_path)
    img = Image.fromarray(pred)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

@app.get("/compare/{model_name}/{epoch}")
async def compare(model_name: str, epoch: int):
    img_str = load_prediction(model_name, epoch)
    if img_str:
        return {"model": model_name, "epoch": epoch, "prediction": img_str}
    return {"error": "Epoch data not found"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
