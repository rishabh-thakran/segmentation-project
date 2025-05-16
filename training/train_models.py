import os
import glob
import torch
import numpy as np
import time
from tqdm import tqdm  # ✅ Progress Bar for Training Visualization
from training.dataset import train_loader, val_loader, test_loader  # ✅ Import Dataset Loaders
from models.unet import UNet
from models.deeplabv3 import DeepLabV3
from models.pspnet import PSPNet
from models.segnet import SegNet
from models.fcn import FCN

# ✅ Auto-detect GPU or fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ✅ Utilize both GPUs if available
multi_gpu = torch.cuda.device_count() > 1
if multi_gpu:
    print(f"🔥 Using {torch.cuda.device_count()} GPUs!")

# ✅ Define segmentation models
models = {
    "unet": UNet(num_classes=150),
    "deeplabv3": DeepLabV3(num_classes=150),
    "pspnet": PSPNet(num_classes=150),
    "segnet": SegNet(num_classes=150),
    "fcn": FCN(num_classes=150)
}

# ✅ Apply `torch.nn.DataParallel` for multi-GPU execution
for model_name in models:
    models[model_name] = models[model_name].to(device)
    if multi_gpu:
        models[model_name] = torch.nn.DataParallel(models[model_name])


# ✅ Define Save Checkpoints & Directories
save_epochs = [5, 10, 20, 50]
save_dir = "saved_predictions"
os.makedirs(save_dir, exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# ✅ Training loop with automatic checkpoint loading
for model_name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    checkpoint_files = sorted(glob.glob(f"saved_models/{model_name}_epoch_*.pth"))
    checkpoint_path = checkpoint_files[-1] if checkpoint_files else f"saved_models/{model_name}.pth"

    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming training for {model_name} from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        last_epoch = int(checkpoint_path.split("_epoch_")[-1].split(".pth")[0]) if "_epoch_" in checkpoint_path else 0
    else:
        print(f"🚀 Starting fresh training for {model_name}")
        last_epoch = 0

    for epoch in range(last_epoch + 1, 51):
        model.train()

        loop = tqdm(train_loader, desc=f"🔥 Training {model_name} - Epoch {epoch}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = torch.nn.CrossEntropyLoss()(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        print(f"✅ Completed epoch {epoch} for {model_name}")

        torch.cuda.empty_cache()

        if epoch in save_epochs:
            model.eval()
            with torch.no_grad():
                sample_image = images[0].unsqueeze(0)
                pred = torch.argmax(model(sample_image), dim=1).squeeze().cpu().numpy().astype(np.uint8)
                np.save(os.path.join(save_dir, f"{model_name}_epoch_{epoch}.npy"), pred)

        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"saved_models/{model_name}_epoch_{epoch}.pth")

           
    torch.save(model.state_dict(), f"saved_models/{model_name}.pth")

print("✅ Training complete! All models saved successfully. 🚀🔥")
