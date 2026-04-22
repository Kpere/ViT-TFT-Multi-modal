import cv2
import numpy as np
from skimage.feature import hog

def extract_candlestick_features(image_paths):
    """
    Extracts manual Computer Vision features (HOG, Color histograms, and edge contours)
    from the generated candlestick charts.
    """
    print("✅ Extracting Hand-crafted Candlestick Features (HOG & Contours)...")
    candlestick_features = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # HOG Features on RGB
        hog_features = hog(img_rgb, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True, channel_axis=-1)
        
        # Color Histogram
        hist_features = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).flatten()
        
        # Body Color (1 if predominantly green/white, 0 if red/black based on channel dist)
        body_color = 1 if np.mean(img[img[:, :, 1] > img[:, :, 2]]) > 127 else 0
        
        # Edge Detection
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        body_size, upper_wick, lower_wick = 0, 0, 0
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h > 5:
                body_size = max(body_size, h)
                if y < img.shape[0] // 3:
                    upper_wick = max(upper_wick, h)
                elif y > img.shape[0] * 2 // 3:
                    lower_wick = max(lower_wick, h)
                    
        # Concat individual engineered features
        feature_vec = np.concatenate([
            [body_color, np.log1p(body_size + 1), np.log1p(upper_wick + 1), np.log1p(lower_wick + 1)], 
            hog_features, 
            hist_features
        ])
        candlestick_features.append(feature_vec)
        
    candlestick_features = np.array(candlestick_features)
    print(f"✅ Candlestick engineered shape: {candlestick_features.shape}")
    
    return candlestick_features
