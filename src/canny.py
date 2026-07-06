import cv2
import numpy as np
from PIL import Image

def cannyPredict(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(img_gray, threshold1=60, threshold2=120, apertureSize=3, L2gradient=True)

    y_coords, x_coords = np.where(canny==255)

    y_counts = np.bincount(y_coords)

    LineThreshold = 25

    bad_y_levels = np.where(y_counts > LineThreshold)[0]

    for bad_y in bad_y_levels:
        canny[bad_y, :]=0
        
    pred_image = Image.fromarray(canny, mode="L")
    return pred_image

if __name__ == "__main__":
    cannyPredict()