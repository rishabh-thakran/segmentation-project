import sys
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QComboBox
from PyQt5.QtGui import QPixmap

# **Import models from specified directory**
sys.path.append(r"C:\Users\trish\OneDrive\Desktop\New folder\Segmentation_Project_Multiple_Models\models")
from unet import UNet
from deeplabv3 import DeepLabV3

# **Set device to GPU (if available)**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# **Model paths**
model_paths = {
    "U-Net": r"C:\Users\trish\OneDrive\Desktop\New folder\Segmentation_Project_Multiple_Models\saved_models\unet_epoch_50.pth",
    "DeepLabV3": r"C:\Users\trish\OneDrive\Desktop\New folder\Segmentation_Project_Multiple_Models\saved_models\deeplabv3_epoch_25.pth"
}

# **Load models properly (initialize first, then load weights)**
def load_model(model_name):
    if model_name == "U-Net":
        model = UNet(num_classes=150)  # ✅ Set correct num_classes
    elif model_name == "DeepLabV3":
        model = DeepLabV3(num_classes=150)  # ✅ Set correct num_classes

    # **Load saved weights, adjusting for multi-GPU training**
    state_dict = torch.load(model_paths[model_name], map_location=device)

    # **Remove "module." prefix if model was trained using DataParallel**
    if "module." in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)  # ✅ Load adjusted state dict
    model.to(device)
    model.eval()  # Set model to evaluation mode
    return model

# **Preprocess image for model input**
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

# **Run segmentation inference**
def segment_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)  # Raw model output
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Multi-class segmentation
    return predicted_mask

# **Generate colormap for segmentation visualization**
def apply_color_map(segmentation_result):
    colormap = np.random.randint(0, 255, (150, 3))  # Generate random colors for each class
    colored_output = colormap[segmentation_result]  # Map segmentation output to colors
    return colored_output

# **PyQt-Based UI**
class SegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.selected_model = "U-Net"
        self.model = load_model(self.selected_model)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # **Image preview**
        self.image_label = QLabel("Select an image...")
        self.image_label.setFixedSize(256, 256)  # Set preview size
        layout.addWidget(self.image_label)

        # **Dropdown menu for model selection**
        self.model_selector = QComboBox()
        self.model_selector.addItems(["U-Net", "DeepLabV3"])
        self.model_selector.currentTextChanged.connect(self.select_model)
        layout.addWidget(self.model_selector)

        # **Load Image Button**
        self.button = QPushButton("Load Image")
        self.button.clicked.connect(self.load_image)
        layout.addWidget(self.button)

        # **Run Segmentation Button**
        self.segment_button = QPushButton("Run Segmentation")
        self.segment_button.clicked.connect(self.run_segmentation)
        layout.addWidget(self.segment_button)

        self.setLayout(layout)
        self.setWindowTitle("Segmentation UI")
        self.show()

    def select_model(self, model_name):
        self.selected_model = model_name
        self.model = load_model(self.selected_model)  # Load selected model

    def load_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File")
        if self.image_path:
            self.image_label.setPixmap(QPixmap(self.image_path).scaled(256, 256))  # **Show image preview**

    def run_segmentation(self):
        if self.image_path:
            try:
                image_tensor = preprocess_image(self.image_path)
                segmentation_result = segment_image(self.model, image_tensor)
                colored_output = apply_color_map(segmentation_result)

                # **Display results**
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(Image.open(self.image_path))
                axes[0].set_title("Original Image")
                axes[1].imshow(colored_output)
                axes[1].set_title(f"Segmentation ({self.selected_model})")

                plt.show(block=False)
            except Exception as e:
                self.image_label.setText(f"Error: {e}")

# **Run the Application**
app = QApplication(sys.argv)
window = SegmentationApp()
sys.exit(app.exec_())
