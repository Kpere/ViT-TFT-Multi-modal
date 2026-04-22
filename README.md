# Multi-Modal ViT-TFT Stock Prediction Framework

A modular, open-source pipeline for stock price movement classification utilizing a hybrid Vision Transformer (ViT) and Temporal Fusion Transformer (TFT) architecture. 
This codebase processes historical data, formats it into 256x256 candlestick charts, and extracts visual and temporal features for classification.

## Project Structure
- `historical_data.py`: Loads CSV datasets and generates customizable technical indicators.
- `image_generation.py`: Takes the tabular time-series data and visualizes it into individual candlestick images using `mplfinance`.
- `feature_extractor_hog.py`: Extracts classical computer vision features (HOG, Histograms, Contours) from the candlestick images.
- `feature_extractor_vit.py`: Uses `timm` and PyTorch to pass the candlestick images through a pretrained Vision Transformer to extract dense visual embeddings.
- `main.py`: The entry script that links all modules together, and demonstrates the data multi-modal fusion architecture (historical data ready for TFT + image data parsed by ViT).

## Installation

Ensure you have Python 3.8+ installed, then install the dependencies via:

```bash
pip install -r requirements.txt
```

## Quick Start
Provide your CSV stock data (e.g., from Yahoo Finance or Investing.com) in `pd.DataFrame` format containing Open, High, Low, Close, Volume.
Update the path in `main.py` and run:

```bash
python main.py
```

## Published Article
Friday, Ibanga Kpereobong, Sarada Prasanna Pati, and Debahuti Mishra. "A multi-modal approach using a hybrid vision transformer and temporal fusion transformer model for stock price movement classification." IEEE Access (2025). Vol. 13, pp. 127221-127239, 2025, doi: 10.1109/ACCESS.2025.3589063. 
https://ieeexplore.ieee.org/document/11080418/
