import os
import json

def create_test_data_json(base_dir, output_file):
    class_labels = []
    
    for class_label in sorted(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_label)
        if os.path.isdir(class_path): 
            class_labels.append({"class_label": class_label})
    

    with open(output_file, 'w') as f:
        for entry in class_labels:
            f.write(json.dumps(entry) + '\n')

create_test_data_json(
    base_dir=r"C:\Users\DEEPIKA\Documents\OpenCVCode\Animal_split\train",  
    output_file="test_animal.json"  
)
