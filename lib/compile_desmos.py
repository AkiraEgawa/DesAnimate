import os
import glob
import re
import sys

def compile(bezier_folder):
    input_dir = bezier_folder
    output_file = "desmos_animated.txt"
    
    # 1. Gather and sort your text files to preserve the timeline
    search_path = os.path.join(input_dir, "edge_frame_*.txt")
    file_paths = sorted(glob.glob(search_path))
    
    if not file_paths:
        print(f"❌ No bezier text files found in '{input_dir}/'. Run vectorize.py first!")
        return
        
    print(f"🚀 Found {len(file_paths)} frames. Packing curves into single-line lists...")
    
    master_lines = []
    
    # Regex to cleanly capture all 8 control numbers from your existing format:
    # ((1-t)^3*x0 + 3(1-t)^2*t*x1 + 3(1-t)*t^2*x2 + t^3*x3, (1-t)^3*y0 + 3(1-t)^2*t*y1 + 3(1-t)*t^2*y2 + t^3*y3)
    pattern = re.compile(
        r"\(\(1-t\)\^3\*([\d\.-]+) \+ 3\(1-t\)\^2\*t\*([\d\.-]+) \+ 3\(1-t\)\*t\^2\*([\d\.-]+) \+ t\^3\*([\d\.-]+),\s*"
        r"\(1-t\)\^3\*([\d\.-]+) \+ 3\(1-t\)\^2\*t\*([\d\.-]+) \+ 3\(1-t\)\*t\^2\*([\d\.-]+) \+ t\^3\*([\d\.-]+)\)"
    )

    # 2. Process each frame file
    for frame_idx, path in enumerate(file_paths, start=1):
        base_name = os.path.basename(path)
        
        try:
            with open(path, "r") as f:
                lines = f.read().splitlines()
                
            # Lists to aggregate each specific control coordinate component across the entire frame
            x0_list, x1_list, x2_list, x3_list = [], [], [], []
            y0_list, y1_list, y2_list, y3_list = [], [], [], []
            
            for line in lines:
                match = pattern.match(line.strip())
                if match:
                    # Unpack coordinates safely
                    x0, x1, x2, x3, y0, y1, y2, y3 = match.groups()
                    x0_list.append(x0); x1_list.append(x1); x2_list.append(x2); x3_list.append(x3)
                    y0_list.append(y0); y1_list.append(y1); y2_list.append(y2); y3_list.append(y3)
            
            # If the frame has no curves, skip it or append an empty restriction
            if not x0_list:
                continue
                
            # 3. Construct the compressed, single-line array equation string
            # Joins elements into comma-separated strings inside square brackets like: [469,470,472]
            compressed_equation = (
                f"((1-t)^3*[{','.join(x0_list)}] + 3(1-t)^2*t*[{','.join(x1_list)}] + 3(1-t)*t^2*[{','.join(x2_list)}] + t^3*[{','.join(x3_list)}], "
                f"(1-t)^3*[{','.join(y0_list)}] + 3(1-t)^2*t*[{','.join(y1_list)}] + 3(1-t)*t^2*[{','.join(y2_list)}] + t^3*[{','.join(y3_list)}])"
                f"{{f={frame_idx}}}"
            )
            
            master_lines.append(compressed_equation)
            print(f"  ✓ Packed {len(x0_list)} curves into 1 Line for Frame {frame_idx} ← {base_name}")
            
        except Exception as e:
            print(f"❌ Error compiling {base_name}: {e}")
            
    # 4. Save the optimized output
    with open(output_file, "w") as out:
        for line in master_lines:
            out.write(line + "\n")
            
    print(f"\n✨ COMPRESSION COMPLETE!")
    print(f"📊 Original layout: Millions of rows.")
    print(f"🚀 New layout: EXACTLY {len(master_lines)} rows.")
    print(f"💾 Open and copy the contents of: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"USAGE: python {sys.argv[1]} <bezier_folder>")
    compile(sys.argv[1])