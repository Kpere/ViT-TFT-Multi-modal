import os
import io
import mplfinance as mpf
from PIL import Image

def generate_candlestick_image(df, start_idx, window=10, image_dir='./candlestick_images'):
    """
    Generates a single candlestick image from the dataframe for a specific window.
    """
    subset = df.iloc[start_idx:start_idx + window]
    if len(subset) < window:
        return None
        
    fig, ax = mpf.plot(subset, type='candle', style='charles', returnfig=True)
    buf = io.BytesIO()
    
    # Save chart memory buffer
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    
    # Resize and Save via PIL
    image_data = Image.open(buf).convert('RGB').resize((256, 256))  # Adjusted to ViT optimal size
    img_path = f"{image_dir}/candlestick_{start_idx}.jpg"
    image_data.save(img_path, format='JPEG', quality=50)
    
    buf.close()
    return img_path

def generate_all_images(df, window_size=5, output_dir='./candlestick_images'):
    """
    Iterates over the dataframe and generates images into the given directory.
    """
    print(f"Generating candlestick images into {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = []
    
    # Limit for demonstration purposes or large datasets
    for i in range(len(df) - window_size):
        img_file = generate_candlestick_image(df, i, window=window_size, image_dir=output_dir)
        if img_file:
            image_files.append(img_file)
            
    print(f"Generated {len(image_files)} candlestick images.")
    return image_files
