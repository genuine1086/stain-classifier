from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

# ===== 모델 관련 설정 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "coffee_model.pth")

CLASS_NAMES = ["Clean", "Coffee", "Wine"]

# ===== 모델 lazy load =====
model = None

def get_model():
    global model
    if model is None:
        print("Loading model for the first time...")
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, len(CLASS_NAMES))
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        model = m
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "Empty filename", 400

    # Save image
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    # Prepare image
    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    # Lazy load model
    m = get_model()

    with torch.no_grad():
        outputs = m(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        _, pred = torch.max(outputs, 1)
        label = CLASS_NAMES[pred.item()]

    result = {CLASS_NAMES[i]: f"{probs[i]*100:.2f}%" for i in range(len(CLASS_NAMES))}
    return render_template("index.html", result=result, filename=file.filename, label=label)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

