import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)  

text = ["A photo of a dhoni"]
image = Image.open("Images\cat.jpg")

image_input = preprocess(image).unsqueeze(0).to(device)
text_input = clip.tokenize(text).to(device)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_input)

image_features /= image_features.norm(dim=-1, keepdim=True
                                      )
text_features /= text_features.norm(dim=-1, keepdim=True)

similarity = (image_features @ text_features.T).squeeze(0)
print("Similarity score:", similarity.item())


