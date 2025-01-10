import os
import cv2

# Define the source and destination directories
source_dir = 'raw-data'
dest_dir = 'raw-images'

# Ensure the destination directory exists
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# List all video files with .mov and .mp4 extensions (case-insensitive)
video_files = [f for f in os.listdir(source_dir) if os.path.splitext(f)[1].lower() in ['.mov', '.mp4']]

# Process each video file
for video_file in video_files:
    video_path = os.path.join(source_dir, video_file)
    
    # Get the base name of the video file without the extension
    base_name = os.path.splitext(video_file)[0]
    
    # Create a directory for the extracted images of this video
    output_dir = os.path.join(dest_dir, base_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video file {video_file}")
        continue

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            # No more frames to read
            break
        
        # Generate a filename for each frame image
        frame_filename = f"frame{frame_count:05d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        
        # Save the frame as an image
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames from {video_file} to {output_dir}")