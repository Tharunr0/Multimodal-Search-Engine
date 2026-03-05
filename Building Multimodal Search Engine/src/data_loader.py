import json
import os

def get_coco_data(json_path, img_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Map image ID to its filename
    id_to_file = {img['id']: img['file_name'] for img in data['images']}
    
    mapping = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in mapping:
            full_path = os.path.join(img_dir, id_to_file[img_id])
            mapping[img_id] = {"path": full_path, "caption": ann['caption']}
            
    return list(mapping.values())