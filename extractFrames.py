import cv2
import os
import sys

def checkInput():
    if len(sys.argv) != 3:
        print(f"Wrong call: python {sys.argv[0]} <video> <frames>")
        sys.exit()


def extract_frames_helper(video_path, target_fps):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"Failed to open video: {video_path}")
        sys.exit(1)

    os.makedirs("frames", exist_ok=True)
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = target_fps

    interval = max(1, round(source_fps / target_fps))
    frame_index = 0
    saved_index = 0

    # Video has too much black frames, skip the starting part of it
    foundContent = False

    while True:
        ret, frame = capture.read()
        if not ret:
            break

        if not foundContent:
            if frame.max() <= 5:
                frame_index += 1
                continue
        else:
            foundContent = True
            print(f"Video Starts at frame {frame_index}")

        if frame_index % interval == 0:
            filename = os.path.join("frames", f"frame_{saved_index:05d}.png")
            cv2.imwrite(filename, frame)
            saved_index += 1

        frame_index += 1

    capture.release()
    print(f"Saved {saved_index} frames from {video_path} at target rate {target_fps} fps")


def extractFrames():
    checkInput()
    video = sys.argv[1]
    try:
        frames = int(sys.argv[2])
    except ValueError:
        print("frames must be an integer")
        sys.exit(1)

    extract_frames_helper(video, frames)

if __name__ == "__main__":
    extractFrames()