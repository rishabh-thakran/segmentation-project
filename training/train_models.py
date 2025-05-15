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
import threading  # ✅ Runs Git commits in background

# ✅ Auto-detect GPU or fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ✅ Utilize both GPUs if available
if torch.cuda.device_count() > 1:
    print(f"🔥 Using {torch.cuda.device_count()} GPUs!")
    multi_gpu = True
else:
    multi_gpu = False

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
os.makedirs("saved_models", exist_ok=True)  # Ensure model save directory exists

# ✅ Git Auto-Commit Function (Runs in Background)
def auto_commit():
    while True:
        os.chdir("/kaggle/working/segmentation-project")
        os.system("git add .")
        os.system("git commit -m 'Auto-commit: Training progress'")
        os.system("git push https://{token}@github.com/rishabh-thakran/segmentation-project.git main")
        time.sleep(1200)  # Commits every 20 minutes

# ✅ Start Git commit thread
git_thread = threading.Thread(target=auto_commit, daemon=True)
git_thread.start()

# ✅ Training loop with automatic checkpoint loading
for model_name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ✅ Find the latest epoch checkpoint if available
    checkpoint_files = sorted(glob.glob(f"saved_models/{model_name}_epoch_*.pth"))
    checkpoint_path = checkpoint_files[-1] if checkpoint_files else f"saved_models/{model_name}.pth"

    if os.path.exists(checkpoint_path):
        print(f"🔄 Resuming training for {model_name} from {checkpoint_path}")
        
        # ✅ Load checkpoint correctly (adjusts 'module.' prefix if needed)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        new_checkpoint = {}

        if any(key.startswith("module.") for key in checkpoint.keys()):
            print("🔄 Adjusting checkpoint for multi-GPU loading...")
            # ✅ Remove 'module.' prefix if necessary
            for key in checkpoint.keys():
                new_key = key.replace("module.", "")
                new_checkpoint[new_key] = checkpoint[key]
        else:
            print("✅ Loading single-GPU checkpoint...")
            new_checkpoint = checkpoint  # No adjustment needed

        # ✅ Load adjusted checkpoint into model
        model.load_state_dict(new_checkpoint, strict=False)
        
        last_epoch = int(checkpoint_path.split("_epoch_")[-1].split(".pth")[0]) if "_epoch_" in checkpoint_path else 0
    else:
        print(f"🚀 Starting fresh training for {model_name}")
        last_epoch = 0

    for epoch in range(last_epoch + 1, 51):  # ✅ Resume training from last saved epoch
        model.train()

        # ✅ Display progress bar for each epoch
        loop = tqdm(train_loader, desc=f"🔥 Training {model_name} - Epoch {epoch}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)  # ✅ Move data to GPU/CPU
            output = model(images)
            loss = torch.nn.CrossEntropyLoss()(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())  # ✅ Show live loss updates

        print(f"✅ Completed epoch {epoch} for {model_name}")

        # ✅ Free GPU memory after each epoch
        torch.cuda.empty_cache()

        # ✅ Save model predictions at specific epochs
        if epoch in save_epochs:
            model.eval()
            with torch.no_grad():
                sample_image = images[0].unsqueeze(0)  # ✅ Select one sample image
                pred = torch.argmax(model(sample_image), dim=1).squeeze().cpu().numpy().astype(np.uint8)
                np.save(os.path.join(save_dir, f"{model_name}_epoch_{epoch}.npy"), pred)

        # ✅ Save model checkpoint every 10 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"saved_models/{model_name}_epoch_{epoch}.pth")

    # ✅ Save final trained model weights
    torch.save(model.state_dict(), f"saved_models/{model_name}.pth")

print("✅ Training complete! All models saved successfully. 🚀🔥")
