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
    pred_mask = probabilities.squeeze(0).squeeze(0).cpu().numpy()
    print(f"→ Processing values -> Min: {pred_mask.min():.4f} | Max: {pred_mask.max():.4f}")
    
    # Create a single unified variable to hold our binary edge mask
    binary_mask = None

    # Path A: Saturated handling (Percentile-based)
    if pred_mask.min() > 0.5:
        print("  ⚠ Saturated confidence detected. Isolating the top 2% sharpest contrast changes...")
        threshold_value = np.percentile(pred_mask, 98)
        binary_mask = (pred_mask >= threshold_value).astype('uint8') * 255
    
    # Path B: Normal / Bad Apple handling (Canny-based)
    else:
        if pred_mask.max() - pred_mask.min() > 1e-5:
            normalized = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min())
        else:
            normalized = pred_mask
        ai_grayscale = (normalized * 255).astype('uint8')
        
        # Use Canny directly to track crisp, sharp lines
        binary_mask = cv2.Canny(ai_grayscale, threshold1=50, threshold2=150)

    # --- UNIFIED MORPHOLOGICAL CLEANING ---
    kernel = np.ones((7, 7), np.uint8)
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # 3. HIERARCHY ISOLATION: RETR_CCOMP pulls both outer and inner boundaries and links them
    contours, hierarchy = cv2.findContours(cleaned_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    final_edges = np.zeros_like(cleaned_mask)
    h, w = cleaned_mask.shape

    # Safety check: make sure contours were actually found
    if hierarchy is not None:
        hierarchy = hierarchy[0] # Flatten the outer hierarchy array
        
        for idx, cnt in enumerate(contours):
            # --- NEW: CONCENTRIC LAYER FILTER ---
            # hierarchy[idx][3] looks at the "Parent" index of the current contour.
            # If it has a parent (value is NOT -1), it means this is a nested inner edge.
            # Skip it to destroy the duplicate/inside lines!
            if hierarchy[idx][3] != -1:
                continue

            # ARTIFACT GUARD: Skip full-frame border artifacts
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            if box_w >= w - 2 or box_h >= h - 2:
                continue  
                
            # NOISE FILTER: Discard tiny pixel specks
            area = cv2.contourArea(cnt)
            min_area = 10 if pred_mask.min() > 0.5 else 30
            if area < min_area: 
                continue
                
            # CONTOURS SMOOTHING (ANTI-ALIASING)
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.006 * perimeter
            smoothed_cnt = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Draw ONLY the verified true parent outer boundary
            cv2.drawContours(final_edges, [smoothed_cnt], -1, (255), thickness=1)
    pred_image = Image.fromarray(final_edges, mode="L")
    pred_image.save(output_path)
    print(f"✓ Post-processing complete. Saved to: {output_path}")

if __name__ == "__main__":
    # Let's test it on a random image from your validation set!
    TEST_IMG = "datasets/images/train/sddefault.jpg"
    OUTPUT_IMG = "output_edges.png"
    
    if os.path.exists(TEST_IMG):
        predict_single_image(TEST_IMG, OUTPUT_IMG)
    else:
        print(f"Could not find test image at {TEST_IMG}. Please swap out the path to any valid image file!")