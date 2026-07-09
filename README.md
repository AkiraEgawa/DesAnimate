# Desmos Vector Engine (Desanimate)

## Overview

The script executes a multi-step pipeline that extracts frames from a video, processes them into lines, converts those lines into Bezier curves, and compiles them into Desmos equations

### Inspiration

I watched a few too many videos of people making art in Desmos, and wanted to see if I could create a project to turn videos into Desmos commands.
Some of the videos that inspired me are:
[Bad Apple Desmos](https://www.youtube.com/watch?v=MVrNn5TuMkY)
[Desmos Final Boss](https://www.youtube.com/watch?v=Q0mqWGRF6aQ)

## Requirements

This program runs on python 3, you can check via `python --version`, it should output a number starting with 3
Additional libraries needed are opencv-python, numpy, and pillow
Installation Process is:
```bash
pip install opencv-python numpy pillow
```

## How to Use

```bash
python main.py <step_number_or_command> <additional_arguments>
```

| Step | Argument | Command Example | Description |
| :--- | :--- | :--- | :--- |
| **Step 0** | `<video_path> <fps>` | `python main.py 0 input.mp4 30` | Starts from the very beginning. Extracts frames from the video at the specified FPS and runs the entire pipeline. |
| **Step 1** | `<frames_folder>` | `python main.py 1 ./frames` | Skips video extraction. Processes pre-extracted image frames into lines. |
| **Step 2** | `<lines_folder>` | `python main.py 2 ./processed_lines` | Skips line extraction. Converts pre-processed line data into Bézier curves. |
| **Step 3** | `<bezier_folder>` | `python main.py 3 ./bezier` | Skips curve generation. Compiles existing Bézier data into Desmos text files. |

Note: The final `desmos_animated.txt` was too large to upload, so to see the bad apple animated, you will need to run the code given in step 3 first to generate it before following the instructions below.

# Playing the Video

Due to the video being too intensive for the web version of Desmos, I opted to use the Desmos API version.
```bash
python -m http.server 8000
```
Then just visit localhost:8000/

# Utility Commands

View Help Menu
```Bash
python desanimate.py help
```

Cleanup Generated outputs (deleltes all generated files and folders for a clean run)
```Bash
python desanimate.py clean
```