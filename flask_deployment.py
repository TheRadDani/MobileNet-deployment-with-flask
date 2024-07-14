import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

def transform_image(image):
  input_transforms = [
      transforms.Resize(255),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
  ]
  my_transform = transforms.Compose(input_transforms)
  image = Image.open(image)
  transformed_image = my_transform(image)
  transformed_image.unsqueeze_(0)
  return transformed_image

def prediction(input):
  outputs = model.forward(input)
  _, predicted = outputs.max(1)
  prediction = predicted.item()
  return prediction

def render_prediction(prediction_idx):
  stridx = str(prediction_idx)
  class_name = "Unknown"
  if img_class_map is not None:
    if stridx in img_class_map is not None:
      class_name = img_class_map[stridx][1]
  return prediction_idx, class_name

import io
import json
import os

app = Flask(__name__)

model = models.densenet121(pretrained=True)
model.eval()

img_class_map = None
mapping_file_path = 'index_to_name.json'

if os.path.isfile(mapping_file_path):
  with open(mapping_file_path) as f:
    img_class_map = json.load(f)

@app.route('/', methods=['GET'])
def root():
  return jsonify({'msg': 'POST to the /predict endpoint with RGB image'})

@app.route('/predict', methods=['POST'])
def predict():
  if request.method == 'POST':
    file = request.files['file']
    if file is not None:
      input_tensor = transform_image(file)
      prediction_idx = prediction(input_tensor)
      prediction_idx, class_name = render_prediction(prediction_idx)
      return jsonify({'prediction_idx': prediction_idx, 'class_name': class_name})

  if __name__ == '__main__':
    app.run()