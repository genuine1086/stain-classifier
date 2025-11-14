from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image
import os

app = Flask(__name__)

# ===== 모델 관련 설정 =====
# app.py 파일이 있는 폴더 기준으로 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "coffee_model.pth")   # 상대경로

CLASS_NAMES = ["Clean", "Coffee", "Wine"]

# 모델 불러오기
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# static 폴더 절대경로
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

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = Image.open(path).convert("RGB")
    img_t = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.softmax(outputs, dim=1)[0]
        _, pred = torch.max(outputs, 1)
        label = CLASS_NAMES[pred.item()]

    result = {CLASS_NAMES[i]: f"{probs[i]*100:.2f}%" for i in range(len(CLASS_NAMES))}
    return render_template("index.html", result=result, filename=file.filename, label=label)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

