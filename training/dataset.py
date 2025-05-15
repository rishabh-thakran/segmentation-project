import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# ✅ Import dataset parsing functions correctly
from training.parse_files import parse_train_val_file, parse_test_file

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, annotations_dir, image_list, image_ext='.jpg', mask_ext='.png', transform=None):
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.image_list = image_list
        self.image_ext = image_ext
        self.mask_ext = mask_ext
        self.transform = transform
        self.mask_transform = transforms.Resize((256, 256))  # Resize masks to ensure consistent dimensions

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.image_dir, image_name + self.image_ext if not image_name.endswith(self.image_ext) else image_name)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        
        if self.annotations_dir:
            mask_name = os.path.splitext(image_name)[0]  # Remove extension from image name
            mask_path = os.path.join(self.annotations_dir, mask_name + self.mask_ext)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"File not found: {mask_path}")
            
            mask = Image.open(mask_path).convert("L")  # Load mask as grayscale
            mask = self.mask_transform(mask)  # Resize mask to standard size
            mask = torch.tensor(np.array(mask), dtype=torch.long)  # Convert mask to tensor with class indices

            # Handle invalid mask values
            mask[mask >= 150] = 0

        else:
            mask = None
        
        if self.transform:
            image = self.transform(image)
        
        return (image, mask) if mask is not None else (image, image_name)

# ✅ Define dataset transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ Load image lists using `parse_files.py`
train_images, train_categories = parse_train_val_file('data/train.txt')
val_images, val_categories = parse_train_val_file('data/val.txt')
test_images = parse_test_file('data/test.txt')

# ✅ Initialize datasets
train_dataset = SegmentationDataset(
    image_dir='data/images/training',
    annotations_dir='data/annotations/training',
    image_list=train_images,
    transform=transform
)

val_dataset = SegmentationDataset(
    image_dir='data/images/validation',
    annotations_dir='data/annotations/validation',
    image_list=val_images,
    transform=transform
)

test_dataset = SegmentationDataset(
    image_dir='data/images/testing',
    annotations_dir=None,  # Test set might not have annotations
    image_list=test_images,
    transform=transform
)

# ✅ Convert datasets into PyTorch dataloaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=8, pin_memory=True)

# Example usage: Print dataset sizes
print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
