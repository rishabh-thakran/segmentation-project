import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from data.parse_files import parse_train_val_file, parse_test_file  # Import the necessary functions

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, annotations_dir, image_list, image_ext='.jpg', mask_ext='.png', transform=None):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.image_list = image_list
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.transform = transform
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        # Check if image_name already has the extension
        if image_name.endswith(self.image_ext):
            image_path = os.path.join(self.image_dir, image_name)
        else:
            image_path = os.path.join(self.image_dir, image_name + self.image_ext)
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        if self.annotations_dir:
            mask_path = os.path.join(self.annotations_dir, image_name + self.mask_ext)
            mask = Image.open(mask_path).convert("L")
        else:
            mask = None
        
        if self.transform:
            image = self.transform(image)
            if mask:
                mask = self.transform(mask)
        
        return (image, image_name) if mask is None else (image, mask)

# Define the transform to resize images and masks to a fixed size
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Example usage for training and validation datasets
train_images, train_categories = parse_train_val_file('data/train.txt')
val_images, val_categories = parse_train_val_file('data/val.txt')
test_images = parse_test_file('data/test.txt')

train_dataset = SegmentationDataset(image_dir='data/images/training',
                                    annotations_dir='data/annotations/training',
                                    image_list=train_images,
                                    image_ext='.jpg',
                                    mask_ext='.png',
                                    transform=transform)

val_dataset = SegmentationDataset(image_dir='data/images/validation',
                                  annotations_dir='data/annotations/validation',
                                  image_list=val_images,
                                  image_ext='.jpg',
                                  mask_ext='.png',
                                  transform=transform)

test_dataset = SegmentationDataset(image_dir='data/testing',  # Correct directory
                                   annotations_dir=None,  # No annotations for test data
                                   image_list=test_images,
                                   image_ext='.jpg',
                                   mask_ext='.png',
                                   transform=transform)
