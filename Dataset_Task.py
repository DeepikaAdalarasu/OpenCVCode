import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(base_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Train, validation, and test ratios must sum to 1.")
    
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            train_images, temp_images = train_test_split(images, test_size=(1 - train_ratio), random_state=42)
            val_images, test_images = train_test_split(temp_images, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)
            
            
            for split, image_list in [("train", train_images), ("val", val_images), ("test", test_images)]:
                split_dir = os.path.join(output_dir, split, category)
                os.makedirs(split_dir, exist_ok=True)
                for image in image_list:
                    shutil.copy(os.path.join(category_path, image), os.path.join(split_dir, image))
                    
    print(f"Dataset split completed! Training, validation, and test sets are saved in '{output_dir}'.")

split_dataset(
    base_dir=r"C:\Users\DEEPIKA\Documents\OpenCVCode\Animal_folder", 
    output_dir=r"C:\Users\DEEPIKA\Documents\OpenCVCode\Animal_split", 
    train_ratio=0.8, 
    val_ratio=0.1, 
    test_ratio=0.1
)
