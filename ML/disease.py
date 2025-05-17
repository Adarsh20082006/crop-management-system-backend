import io
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset (only for class labels)
dataset = datasets.ImageFolder(r'PlantVillage', transform=transform)

# Split dataset (optional, here just to match your training pipeline)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet18 model with updated weights parameter
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)

# Replace final layer for number of classes in your dataset
num_classes = len(dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer (optional, for training)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Make sure to load your trained model weights here if you have saved them, e.g.:
# model.load_state_dict(torch.load('your_model_weights.pth', map_location=device))

# Prediction function (not used directly in Flask route but handy for testing)
def predict(image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return dataset.classes[predicted.item()]

# Flask route to handle image upload and prediction
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    predicted_class = dataset.classes[predicted.item()]

    # Calculate confidence score
    probs = torch.nn.functional.softmax(output, dim=1)
    confidence = probs[0][predicted.item()].item() * 100

    return jsonify({
        'healthStatus': predicted_class,
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(port=5001,debug=True)