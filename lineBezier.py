import cv2
import numpy as np
import os
import glob

def fit_bezier_handles(p0, p3, scale=0.333):
    """
    Calculates control points P1 and P2 between anchor points P0 and P3.
    For an easy, smooth default, we place the handles 1/3 of the distance
    along the vector connecting the anchors.
    """
    p0 = np.array(p0, dtype=float)
    p3 = np.array(p3, dtype=float)
    
    # Calculate vector between the anchors
    vector = p3 - p0
    
    # Place control handles along the direct path
    p1 = p0 + scale * vector
    p2 = p3 - scale * vector
    
    return p1, p2

def image_to_desmos_bezier(image_path):
    # Load your crisp edge image
    img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    if img is None:
        return []
        
    # Find continuous point chains
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    desmos_equations = []
    
    for cnt in contours:
        # 1. Simplify the pixel chain into geometric anchor keys
        # Epsilon dictates accuracy. 0.01-0.02 keeps curves perfectly smooth for Desmos.
        perimeter = cv2.arcLength(cnt, False)
        if perimeter < 10: # Skip noise
            continue
            
        epsilon = 0.00 * perimeter
        anchors = cv2.approxPolyDP(cnt, epsilon, False).squeeze(1)
        
        # We need at least two points to draw a curve segment
        if len(anchors) < 2:
            continue
            
        # 2. Loop through adjacent pairs of anchors and fit a Bézier curve
        for i in range(len(anchors) - 1):
            p0 = anchors[i]
            p3 = anchors[i+1]
            
            # Invert y-axis math because computer vision images measure (0,0) from the top-left,
            # but Desmos graphs Cartesian layout measuring (0,0) from the bottom-left!
            h, _, _ = img.shape if len(img.shape) == 3 else (img.shape[0], 0, 0)
            p0_y = h - p0[1]
            p3_y = h - p3[1]
            
            p1, p2 = fit_bezier_handles((p0[0], p0_y), (p3[0], p3_y))
            
            # 3. Format into a raw string ready for Desmos parametric tables
            # Formula: ((1-t)^3*x0 + 3(1-t)^2*t*x1 + 3(1-t)*t^2*x2 + t^3*x3, (1-t)^3*y0 + 3(1-t)^2*t*y1 + 3(1-t)*t^2*y2 + t^3*y3)
            equation = (
                f"((1-t)^3*{p0[0]} + 3(1-t)^2*t*{p1[0]:.2f} + 3(1-t)*t^2*{p2[0]:.2f} + t^3*{p3[0]}, "
                f"(1-t)^3*{p0_y} + 3(1-t)^2*t*{p1[1]:.2f} + 3(1-t)*t^2*{p2[1]:.2f} + t^3*{p3_y})"
            )
            desmos_equations.append(equation)
            
    return desmos_equations

if __name__ == "__main__":
    # Test on your processed Canny line frame
    TEST_FRAME = "processed_lines/edge_frame_00090.png"
    TXT_OUTPUT = "desmos_equations.txt"
    
    if os.path.exists(TEST_FRAME):
        print(f"→ Calculating Bézier curves for: {TEST_FRAME}")
        curves = image_to_desmos_bezier(TEST_FRAME)
        
        # 1. Print ABSOLUTELY EVERYTHING to the terminal window
        print(f"\n✨ Generated {len(curves)} Bézier segments!\n")
        for c in curves:
            print(c)
            
        # 2. Save everything to a text file for ultimate ease of copy-pasting
        with open(TXT_OUTPUT, "w") as f:
            for c in curves:
                f.write(c + "\n")
                
        print(f"\n💾 Saved all {len(curves)} equations perfectly to: {TXT_OUTPUT}")
    else:
        print(f"Could not find processed edge map at {TEST_FRAME}. Please check your folder paths!")