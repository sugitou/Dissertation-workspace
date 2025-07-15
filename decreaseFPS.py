import cv2
import os

# Input and output video paths
input_path = "tshirtman.mp4"
output_path = "output_targetfps.mp4"

# Target FPS
target_fps = 8


# Open input video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {input_path}")

# Get original FPS and frame size
original_fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate frame interval to skip frames
frame_interval = int(round(original_fps / target_fps))
if frame_interval < 1:
    frame_interval = 1

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

print(f"Converting {input_path} from {original_fps:.2f} fps to {target_fps} fps...")

# Read frames and write to output video
frame_id = 0
written_frames = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % frame_interval == 0:
        out.write(frame)
        written_frames += 1
    frame_id += 1

cap.release()
out.release()
print(f"Done! Saved {written_frames} frames to {output_path}")
