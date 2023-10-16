from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import cgi
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from io import BytesIO

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if self.path == "/getpic":
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD':'POST',
                         'CONTENT_TYPE':self.headers['Content-Type'],
                         })

            fileitem = form['image']

            if fileitem.filename:
                image = Image.open(BytesIO(fileitem.file.read())).convert('RGB')
                transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])
                image_tensor = transform(image).unsqueeze(0)

                vit = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
                vit.heads = nn.Linear(in_features=768, out_features=2)

                mf = torch.max(torch.softmax(vit(image_tensor), dim=1), 1)[1].item()
                mapp = {0:'dog',1:'cat'}
                result = mapp[mf]

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'result': result}).encode())

httpd = HTTPServer(('localhost', 5000), SimpleHTTPRequestHandler)
httpd.serve_forever()
