import torch
import clip
from PIL import Image
from tqdm import tqdm
import numpy as np

class MultimodalEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def encode_images(self, image_paths, batch_size=64):
        embeddings = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding"):
            batch = []
            for path in image_paths[i : i + batch_size]:
                batch.append(self.preprocess(Image.open(path)))
            
            inputs = torch.stack(batch).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(inputs)
                feat /= feat.norm(dim=-1, keepdim=True)
                embeddings.append(feat.cpu().numpy())
                
        return np.vstack(embeddings).astype('float32')

    def encode_text(self, text):
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu().numpy().astype('float32')