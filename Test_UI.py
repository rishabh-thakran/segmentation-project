import sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QComboBox
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
    "DeepLabV3": r"C:\Users\trish\OneDrive\Desktop\New folder\Segmentation_Project_Multiple_Models\saved_models\deeplabv3_epoch_30.pth"
}

# **Define 150-Class Semantic Labels**
CLASS_LABELS = {
    1: "Wall", 2: "Building", 3: "Sky", 4: "Floor", 5: "Tree",
    # Continue adding all 150 labels...
    150: "Flag"
}

# **Generate a Random Colormap for 150 Classes**
COLORMAP = np.random.randint(0, 255, (151, 3), dtype=np.uint8)

# **Load models properly**
def load_model(model_name):
    if model_name == "U-Net":
        model = UNet(num_classes=150)
    elif model_name == "DeepLabV3":
        model = DeepLabV3(num_classes=150)

    state_dict = torch.load(model_paths[model_name], map_location=device)

    # **Handle multi-GPU trained models**
    if "module." in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# **Preprocess image for model input**
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # **Increased size**
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(device)

# **Run segmentation inference**
def segment_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return predicted_mask

# **Apply Semantic Labels & Color Mapping**
def apply_semantics(segmentation_result):
    color_segmented = COLORMAP[segmentation_result]
    label_positions = {}

    unique_classes = np.unique(segmentation_result)
    for class_id in unique_classes:
        if class_id in CLASS_LABELS:
            y_positions, x_positions = np.where(segmentation_result == class_id)
            if len(y_positions) > 0:
                label_positions[class_id] = (int(np.mean(x_positions)), int(np.mean(y_positions)))

    # **Overlay Labels**
    for class_id, position in label_positions.items():
        cv2.putText(color_segmented, CLASS_LABELS.get(class_id, "Unknown"), position, cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), thickness=3, lineType=cv2.LINE_AA)

    return color_segmented, unique_classes  # Return segmented image + detected class IDs

# **PyQt-Based UI**
class SegmentationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.selected_model = "U-Net"
        self.model = load_model(self.selected_model)
        self.initUI()

    def initUI(self):
        # **Enable Larger Window Size**
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowTitle("Semantic Segmentation UI")

        layout = QHBoxLayout()

        # **Left Panel (Image & Buttons)**
        left_panel = QVBoxLayout()

        self.image_label = QLabel("Select an image...")
        self.image_label.setFixedSize(512, 512)
        left_panel.addWidget(self.image_label)

        self.model_selector = QComboBox()
        self.model_selector.addItems(["U-Net", "DeepLabV3"])
        self.model_selector.setStyleSheet("font-size: 16px; padding: 8px; border-radius: 5px;")
        self.model_selector.currentTextChanged.connect(self.select_model)
        left_panel.addWidget(self.model_selector)

        self.button = QPushButton("Load Image")
        self.button.setStyleSheet("font-size: 18px; padding: 10px; border-radius: 10px;")
        self.button.clicked.connect(self.load_image)
        left_panel.addWidget(self.button)

        self.segment_button = QPushButton("Run Segmentation")
        self.segment_button.setStyleSheet("font-size: 18px; padding: 10px; border-radius: 10px;")
        self.segment_button.clicked.connect(self.run_segmentation)
        left_panel.addWidget(self.segment_button)

        layout.addLayout(left_panel)
        self.setLayout(layout)
        self.showMaximized()

    def select_model(self, model_name):
        self.selected_model = model_name
        self.model = load_model(self.selected_model)

    def load_image(self):
        self.image_path, _ = QFileDialog.getOpenFileName(self, "Open Image File")
        if self.image_path:
            self.image_label.setPixmap(QPixmap(self.image_path).scaled(512, 512))

    def run_segmentation(self):
        if self.image_path:
            try:
                image_tensor = preprocess_image(self.image_path)
                segmentation_result = segment_image(self.model, image_tensor)
                semantic_output, detected_classes = apply_semantics(segmentation_result)

                # **Display results in Matplotlib figure with legend**
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(Image.open(self.image_path))
                axes[0].set_title("Original Image")
                axes[1].imshow(semantic_output)
                axes[1].set_title(f"Semantic Segmentation ({self.selected_model})")

                # **Add Legend**
                legend_patches = [plt.Line2D([0], [0], color=COLORMAP[class_id] / 255, marker='o', markersize=8, linestyle='None', label=CLASS_LABELS[class_id]) for class_id in detected_classes if class_id in CLASS_LABELS]
                axes[1].legend(handles=legend_patches, loc="upper right", fontsize=8, title="Detected Classes")

                plt.show(block=False)
            except Exception as e:
                print(f"Error: {e}")

# **Run the Application**
app = QApplication(sys.argv)
window = SegmentationApp()
sys.exit(app.exec_())
