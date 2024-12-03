from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_paths = [r"Images\dog_bike_car.jpg", r"Images\dog_lying.jpg", r"Images\dogs.jpg"] 

captions = []
for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    captions.append(caption)

for img, caption in zip(image_paths, captions):
    print(f"Image: {img}\nCaption: {caption}\n")
