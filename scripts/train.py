import sys
import os

# Add the project directory to sys.path
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

import yaml
import torch
from torch.utils.data import DataLoader
from data.dataset import SegmentationDataset
from data.parse_files import parse_test_file, parse_train_val_file, read_categories_file
from models.unet import UNet
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path,weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Checkpoint loaded: start_epoch={start_epoch}, best_val_loss={best_val_loss}")
        return start_epoch, best_val_loss
    else:
        print(f"No checkpoint found at: {checkpoint_path}")
        return 0, float('inf')

def train(config, resume_checkpoint=None):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_images, _ = parse_train_val_file(config['train_images'])
    val_images, _ = parse_train_val_file(config['val_images'])
    test_images = parse_test_file(config['test_images'])
    
    transform = transforms.Compose([
        transforms.Resize(tuple(config['image_size'])),
        transforms.ToTensor()
    ])
    
    train_dataset = SegmentationDataset(image_dir=os.path.join(config['image_dir'], 'training'),
                                        annotations_dir=os.path.join(config['annotations_dir'], 'training'),
                                        image_list=train_images,
                                        image_ext=config['image_ext'],
                                        mask_ext=config['mask_ext'],
                                        transform=transform)
    
    val_dataset = SegmentationDataset(image_dir=os.path.join(config['image_dir'], 'validation'),
                                      annotations_dir=os.path.join(config['annotations_dir'], 'validation'),
                                      image_list=val_images,
                                      image_ext=config['image_ext'],
                                      mask_ext=config['mask_ext'],
                                      transform=transform)
    
    test_dataset = SegmentationDataset(image_dir=os.path.join(config['image_dir'], 'testing'),
                                       annotations_dir=os.path.join(config['annotations_dir'], 'validation'),  # Adjust if needed
                                       image_list=test_images,
                                       image_ext=config['image_ext'],
                                       mask_ext=config['mask_ext'],
                                       transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = UNet(num_classes=150).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss().to(device)
    
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_interval = 5
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_checkpoint:
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, resume_checkpoint)

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']} - Training") as pbar:
            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks.squeeze(1)
                
                optimizer.zero_grad()
                outputs = model(images)
                
                loss = criterion(outputs, masks.long())  # Convert masks to long for CrossEntropyLoss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)
        
        model.eval()
        with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{config['epochs']} - Validation") as pbar:
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    masks = masks.squeeze(1)
                    
                    outputs = model(images)

                    loss = criterion(outputs, masks.long())  # Convert masks to long for CrossEntropyLoss
                    
                    val_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(model_dir, f"unet_checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_dir, "unet_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss}")

    final_model_path = os.path.join(model_dir, "unet_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

if __name__ == "__main__":
    with open("config/default_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    resume_checkpoint = "saved_models/unet_checkpoint_epoch_20.pth"  # Update with the checkpoint path if resuming
    train(config, resume_checkpoint)
