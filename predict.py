# predict.py
import os
import torch
from PIL import Image
import torchvision.transforms.v2 as transforms
from src.model import GraphCVNet
import cv2
import numpy as np

def predict_single_image(image_path, output_path, weights_path="weights/graphcv_best.pth"):
    # 1. Setup execution device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Instantiate and load the trained model weights
    model = GraphCVNet()
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval() # Set model to evaluation mode (turns off dropout/batchnorm updates)

    # 3. Load and preprocess the raw image
    raw_image = Image.open(image_path).convert("RGB")
    
    # Match the exact preprocessing resizing we used in training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ])
    
    # Add a fake batch dimension: [3, 256, 256] -> [1, 3, 256, 256]
    image_tensor = transform(raw_image).unsqueeze(0).to(device)

    # 4. Forward pass through the AI
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits) # Convert raw numbers to 0.0 - 1.0 confidence values
        
    # 5. Post-process tensor back into a viewable image
    # Remove batch dimension and scale back to 0-255 integers
    pred_mask = probabilities.squeeze(0).squeeze(0).cpu().numpy()
    print(f"→ Processing values -> Min: {pred_mask.min():.4f} | Max: {pred_mask.max():.4f}")
    
    # 1. DEFENSE: Check if the model is heavily saturated (stuck near 1.0)
    if pred_mask.min() > 0.5:
        print("  ⚠ Saturated confidence detected. Isolating the top 2% sharpest contrast changes...")
        threshold_value = np.percentile(pred_mask, 98)
        binary_mask = (pred_mask >= threshold_value).astype('uint8') * 255
    else:
        # Standard robust normalization for normal, well-behaved ranges
        if pred_mask.max() - pred_mask.min() > 1e-5:
            normalized = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
        else:
            normalized = pred_mask
        binary_mask = (normalized > 0.35).astype('uint8') * 255

    # 2. CLEANING: Use morphological closing to bridge minor gaps
    kernel = np.ones((7, 7), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. SILHOUETTE ISOLATION: cv2.RETR_EXTERNAL ensures we only get the outermost boundaries
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_edges = np.zeros_like(cleaned_mask)
    h, w = cleaned_mask.shape

    for cnt in contours:
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        if box_w >= w - 2 or box_h >= h - 2:
            continue  
            
        area = cv2.contourArea(cnt)
        min_area = 10 if pred_mask.min() > 0.5 else 30
        if area < min_area: 
            continue
            
        # --- NEW: CONTOUR SMOOTHING (ANTI-ALIASING) ---
        # epsilon controls how aggressive the smoothing is. 
        # 0.005 to 0.01 times the contour perimeter is usually the sweet spot!
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.006 * perimeter
        smoothed_cnt = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Draw the smooth, simplified contour path instead of the raw jagged one
        cv2.drawContours(final_edges, [smoothed_cnt], -1, (255), thickness=1)
        
    pred_image = Image.fromarray(final_edges, mode="L")
    pred_image.save(output_path)
    print(f"✓ Saturated defense + Exterior boundary isolation applied. Saved to: {output_path}")

if __name__ == "__main__":
    # Let's test it on a random image from your validation set!
    TEST_IMG = "datasets/images/train/sddefault.jpg"
    OUTPUT_IMG = "output_edges.png"
    
    if os.path.exists(TEST_IMG):
        predict_single_image(TEST_IMG, OUTPUT_IMG)
    else:
        print(f"Could not find test image at {TEST_IMG}. Please swap out the path to any valid image file!")