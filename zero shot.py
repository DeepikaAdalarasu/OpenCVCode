import os
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

categories = ["person", "dog", "cat", "ball"]
image = Image.open(r"C:\Users\DEEPIKA\Documents\OpenCVCode\Images\basketball.jpg")

image_input = preprocess(image).unsqueeze(0).to(device)
text_inputs = clip.tokenize(categories).to(device)
 
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarities = (image_features @ text_features.T).squeeze(0)
best_category_idx = similarities.argmax().item()
print(f"The image is classified as: {categories[best_category_idx]} with similarity score: {similarities[best_category_idx].item()}")
