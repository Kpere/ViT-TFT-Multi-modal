import numpy as np
import torch
import timm
from PIL import Image

def extract_vit_features(image_paths):
    """
    Extracts deep visual embeddings using a pretrained Vision Transformer.
    """
    print("✅ Extracting Deep ViT Tokens...")
    
    # Load Pretrained ViT base
    vit_model = timm.create_model('vit_base_patch16_224', pretrained=True)
    vit_model.eval()
    
    visual_tokens = []
    
    for path in image_paths:
        img = Image.open(path).convert('RGB').resize((224, 224))
        
        # Convert to tensor and scale
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        
        with torch.no_grad():
            vit_features = vit_model(img_tensor).squeeze().numpy()
            
        visual_tokens.append(vit_features)
        
    visual_tokens = np.array(visual_tokens)
    print(f"✅ ViT token shape: {visual_tokens.shape}")
    
    return visual_tokens
