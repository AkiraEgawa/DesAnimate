from lib import *
import sys
import shutil
from pathlib import Path


def help():
    print(f"USAGE: python {sys.argv[0]} step#|help <additional arguments>")
    print("Step # denotes the starting point:")
    print("0 = video, argments = <video> <fps>")
    print("1 = frames, arguments = <frames_folder>")
    print("2 = lines, arguments = <lines_folder>")
    print("3 = beziers, arguments = <bezier_folder>")
    print("4 = equations, arguments = <desmos_equations.text>")
    print(f"To clean the standard outputs: python {sys.argv[0]} clean")

def clean():
    folders_to_delete = ['bezier', 'frames', 'processed_lines']
    files_to_delete = ['desmos_equations.txt', 'desmos_animated.txt']

    for folder_name in folders_to_delete:
        folder_path = Path(folder_name)
        if folder_path.is_dir():
            shutil.rmtree(folder_path)
    for file_name in files_to_delete:
        file_path = Path(file_name)
        if file_path.is_file():
            file_path.unlink()
    print("Cleanup Complete")

def main():
    """
    Running this should hopefully run the entire sequence after determining the step
    """
    if len(sys.argv) < 2:
        help()
        exit()
    if sys.argv[1] == "help":
        help()
        exit()
    if sys.argv[1] == "clean":
        clean()
        exit()

    # Okay, so we have an actual one
    pipeline = [extract_frames_helper, extractLines, lineToBezier, compile]

    if ((sys.argv[1] == "0") and (len(sys.argv) != 4)) or ((sys.argv[1]!="0") and (len(sys.argv) != 3)) or not sys.argv[1].isdigit():
        help()
        exit()
    
    frames = "frames"
    processed_lines = "processed_lines"
    bezier = "bezier"
    equations = "desmos_equations.txt"
    target_fps = 0
    video = ""

    step = int(sys.argv[1])

    match step:
        case 0:
            video = sys.argv[2]
            target_fps = int(sys.argv[3])
        case 1:
            frames = sys.argv[2]
        case 2:
            processed_lines = sys.argv[2]
        case 3:
            bezier = sys.argv[2]
        case 4:
            equations = sys.argv[2]
    
    inputs = [(video, target_fps), frames, processed_lines, bezier, equations]
    
    for i, func in enumerate(pipeline[step:]):
        # runs everything sequentially
        index = i+step
        func(inputs[index])

if __name__=="__main__":
    main()