import torch
from models.unet import UNet

# Load the trained model
model = UNet(num_classes=150)
checkpoint = torch.load('saved_models/unet_final.pth')
model.load_state_dict(checkpoint)

# Export the model to ONNX
dummy_input = torch.randn(1, 3, 256, 256)  # Adjust the input size as needed
torch.onnx.export(model, dummy_input, 'saved_models/unet_final.onnx', input_names=['input'], output_names=['output'])

print("Model has been successfully exported to ONNX format.")
