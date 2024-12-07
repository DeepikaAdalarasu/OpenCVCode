import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import json

class CustomDataset(Dataset):
    def __init__(self, json_file, image_dir, transform=None):
        """
        Args:

            json_file (str): Path to JSON file with image paths and captions.
            image_dir (str): Path to the directory containing images.
            transform (callable, optional): Optional transform to apply to images.
        """
        with open(json_file, 'r') as f:
            self.data = [json.loads(line) for line in f.readlines()]
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = os.path.join(self.image_dir, entry['image_path'])
        caption = entry['caption']

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, caption
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
])


dataset = CustomDataset(
    json_file=r"C:\Users\DEEPIKA\Documents\OpenCVCode\predicted_animal.json",  
    image_dir=r"C:\Users\DEEPIKA\Documents\OpenCVCode\Animal_split\train",  
    transform=transform,
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(5):  
    model.train()
    for images, captions in dataloader:
        inputs = processor(text=captions, images=images, return_tensors="pt", do_rescale=False).to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        labels = torch.arange(len(images)).to(device)

        image_loss = loss_fn(logits_per_image, labels)
        text_loss = loss_fn(logits_per_text, labels) 
        loss = (image_loss + text_loss) / 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}: Loss = {loss.item()}")

model.save_pretrained("fine_tuned_clip_model")
processor.save_pretrained("fine_tuned_clip_processor")

def predict(image_path, text_list):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=text_list, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()
    return {text: prob for text, prob in zip(text_list, probs[0])}

result = predict(r"C:\Users\DEEPIKA\Documents\OpenCVCode\Animal_split\train\lion\0100.jpg", ["lion", "dog", "tiger"])     
print(result)

