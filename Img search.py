import os
import clip
import torch
from PIL import Image


device="cuda" if torch.cuda.is_available() else "cpu"
model,preprocess=clip.load("ViT-B/32",device)

text_query="Group of dogs"
image_folder=r"C:\Users\DEEPIKA\Documents\OpenCVCode\Images"

text_input=clip.tokenize([text_query]).to(device)

image_paths=[os.path.join(image_folder,img) for img in os.listdir(image_folder)]
similarities=[]

for img_path in image_paths:
    image=Image.open(img_path)
    image_input=preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    similarity = (image_features @ text_features.T).item()

    similarities.append((img_path, similarity))

best_match = max(similarities, key=lambda x: x[1])
print(f"Best match image: {best_match[0]} with similarity score: {best_match[1]}")