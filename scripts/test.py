import sys
import os

# Add the project directory to sys.path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import SegmentationDataset
from data.parse_files import parse_test_file
from models.unet import UNet
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_model(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(model_state_dict)
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        raise FileNotFoundError(f"No checkpoint found at: {checkpoint_path}")

def save_prediction(prediction, image_name, output_dir):
    prediction = prediction.squeeze().cpu().numpy().astype(np.uint8)
    prediction_image = Image.fromarray(prediction)
    output_path = os.path.join(output_dir, image_name.replace(".jpg", ".png"))
    prediction_image.save(output_path)

def visualize_predictions(test_dir, predictions_dir, image_names, num_examples=5):
    fig, axes = plt.subplots(num_examples, 2, figsize=(10, num_examples * 5))

    for i in range(num_examples):
        # Load original image
        original_image_path = os.path.join(test_dir, image_names[i])
        image = Image.open(original_image_path)
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f"Original Image: {image_names[i]}")

        # Load prediction
        prediction_image_path = os.path.join(predictions_dir, image_names[i].replace(".jpg", ".png"))
        prediction_image = Image.open(prediction_image_path)
        axes[i, 1].imshow(prediction_image, cmap='gray')
        axes[i, 1].set_title(f"Prediction: {image_names[i]}")

    plt.tight_layout()
    plt.savefig(os.path.join(predictions_dir, 'predictions_visualization.png'))
    plt.show()

def evaluate_model(config, model_checkpoint):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_images = parse_test_file(config['test_images'])
    print(f"Number of test images: {len(test_images)}")

    transform = transforms.Compose([
        transforms.Resize(tuple(config['image_size'])),
        transforms.ToTensor()
    ])

    test_dataset = SegmentationDataset(image_dir='data/testing',  # Correct directory
                                       annotations_dir=None,  # No annotations for test data
                                       image_list=test_images,
                                       image_ext=config['image_ext'],
                                       mask_ext=config['mask_ext'],
                                       transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = UNet(num_classes=150).to(device)  # Update num_classes to 150
    load_model(model, model_checkpoint)
    model.eval()

    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)

    all_preds = []
    image_names = []

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc="Testing") as pbar:
            for images, img_names in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

                for pred, img_name in zip(preds, img_names):
                    save_prediction(pred, img_name, output_dir)
                    all_preds.append(pred.cpu().numpy())
                    image_names.append(img_name)
                
                pbar.update(1)

    # Visualize predictions
    visualize_predictions('data/testing', output_dir, image_names)

    print(f"Predictions saved to {output_dir}")

if __name__ == "__main__":
    with open("config/default_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    model_checkpoint = "saved_models/unet_final.pth"  # Update with the best model checkpoint path
    evaluate_model(config, model_checkpoint)
