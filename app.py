from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# =========================
# 1️⃣ Model Setup (MATCH TRAINING)
# =========================

NUM_CLASSES = 3

# Load pretrained ResNet18 (same as training)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze backbone (same as training)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load trained weights
model.load_state_dict(torch.load("walkability_model.pt", map_location="cpu"))

model.eval()

# =========================
# 2️⃣ Image Transform (MATCH TRAINING)
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# 3️⃣ Class Names (MATCH TRAINING)
# =========================

classes = [
    "Moderate",
    "Safe",
    "Unsafe"
]

# =========================
# Upload Folder Config
# =========================

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# Routes
# =========================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None
    confidence = None
    error = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "No file uploaded."
            return render_template("index.html", error=error)

        file = request.files["image"]

        if file.filename == "":
            error = "Please select an image."
            return render_template("index.html", error=error)

        try:
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            # Open and preprocess image
            image = Image.open(save_path).convert("RGB")
            image = transform(image).unsqueeze(0)

            # Prediction
            with torch.no_grad():
                output = model(image)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                conf, predicted = torch.max(probabilities, 1)

                prediction = classes[predicted.item()]
                confidence = round(conf.item() * 100, 2)

            image_path = save_path

        except Exception as e:
            error = f"Error processing image: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        error=error
    )


# =========================
# Run App
# =========================

if __name__ == "__main__":
    app.run(debug=True)
