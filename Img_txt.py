import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = "dog_lying.jpg"
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

descriptions = [
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a Dhoni",
    "a painting of a landscape"
]

text_inputs = clip.tokenize(descriptions).to(device)

image_features = model.encode_image(image)
image_features /= image_features.norm(dim=-1, keepdim=True)

text_features = model.encode_text(text_inputs)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarities = (image_features @ text_features.T).squeeze(0)
best_idx = similarities.argmax().item()

print(f"Best Description: {descriptions[best_idx]}")
    