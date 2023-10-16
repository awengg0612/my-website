import torch,torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)

vit.heads =nn.Linear(in_features=768,out_features=2)
model = vit
transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)
        ]) # test transform
image = transform(Image.open('F:\貓咪.jpg').convert('RGB')).unsqueeze(0)
image = torch.randn(1,3,224,224)
mf = torch.max(torch.softmax(model(torch.randn(1,3,224,224)), dim = 1), 1)[1].item()
mapp = {0:'dog',1:'cat'}
print(mapp[mf])
