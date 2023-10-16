from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)  # Handle CORS

# Load the pre-trained model and modify the last layer
vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
vit.heads = torch.nn.Linear(in_features=768, out_features=2)
vit.eval()  # Set the model to evaluation mode

vit = torch.load("F:/flask/login_project/login_app/0/0.pth",map_location=torch.device('cpu'))

# Ensure you load your custom trained weights
# vit.load_state_dict(torch.load('path_to_your_custom_weights.pth'))

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route('/getpic', methods=['POST'])
def getpic():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    imagefile = request.files['image']
    if imagefile.filename == '':
        return jsonify({"error": "No image selected"}), 400

    img = Image.open(imagefile).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    output = vit(img_t)
    _, predicted = torch.max(output.data, 1)
    mapp = {0: 'cat', 1: 'dog'}  # Ensure this mapping is consistent with your training data
    return jsonify({'result': mapp[predicted.item()]})

if __name__ == '__main__':
    app.run(port=5000)
