from src.canny import cannyPredict
import os
import glob
from PIL import Image
import cv2
import sys

def extractLines(input_dir):
    output_dir = "processed_lines"

    os.makedirs(output_dir, exist_ok=True)

    search_path = os.path.join(input_dir, "frame_*.png")
    frame_paths = sorted(glob.glob(search_path))

    if not frame_paths:
        print(f"No frames found in {input_dir}/, run extractFrames.py first")
        return
    
    print(f"Path found")

    for path in frame_paths:
        base_name = os.path.basename(path)

        try:
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                print(f"Could not read image {base_name}")
                continue
            edge_output = cannyPredict(img_bgr)

            output_path = os.path.join(output_dir, f"edge_{base_name}")

            if isinstance(edge_output, Image.Image):
                edge_output.save(output_path)
            else:
                Image.fromarray(edge_output).convert("L").save(output_path)
            
        except Exception as e:
            print(f"Ran into Error: {e}")
        
    print(f"All images processed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"USAGE: python {sys.argv[0]} <frames>")
        exit()
    input_dir = sys.argv[1]
    extractLines(input_dir)