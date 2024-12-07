import os
import json
import torch
from PIL import Image
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

model = resnet18(weights=ResNet18_Weights.DEFAULT)


def load_class_labels(json_file):
    class_labels = {}
    with open(json_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            data = json.loads(line)
            class_labels[idx] = data['class_label']  
    return class_labels

def predict_image(image_path, model, transform, class_labels):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  

    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)  
    
    if predicted_idx.item() >= len(class_labels):
        raise ValueError(f"Predicted index {predicted_idx.item()} is out of range for the class labels.")
    
    predicted_class = class_labels[predicted_idx.item()]
    return predicted_class

def create_animal_json_with_predictions(base_dir, output_file, json_file):
    dataset = []
    
    class_labels = load_class_labels(json_file)
    
    model = models.resnet18(pretrained=True)
    num_classes = len(class_labels)  
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.eval() 
    

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    for class_label in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_label)
        if os.path.isdir(class_path):  
            for image_name in os.listdir(class_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):  
                    image_path = os.path.join(class_path, image_name)
                    
                    predicted_class = predict_image(image_path, model, transform, class_labels)
                    
                    dataset.append({
                        "image_url": "",  
                        "image_path": image_path.replace("\\", "/"),  
                        "class_label": predicted_class,
                        "caption": f"Image shows that {predicted_class}"
                    })

    with open(output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')

create_animal_json_with_predictions(
    base_dir=r"C:\Users\DEEPIKA\Documents\OpenCVCode\Animal_split\test", 
    output_file="predicted_animal.json",
    json_file=r"C:\Users\DEEPIKA\Documents\OpenCVCode\test_animal.json"  # Path to your original training JSON
)
