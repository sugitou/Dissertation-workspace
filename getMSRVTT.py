import os
import pandas as pd
import yt_dlp

from moviepy.video.io.VideoFileClip import VideoFileClip
from datasets import load_dataset


# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("friedrichor/MSR-VTT", "test_1k")
print(ds)

df = ds["test"].to_pandas()
print(df.head())


# Make the save directory
save_dir = "msrvtt_clips"
os.makedirs(save_dir, exist_ok=True)

# Try to download and trim the first 3 videos
sample_df = df.head(3)

for index, row in sample_df.iterrows():
    video_id = row["id"]
    caption = row["caption"]
    category = row["category"]
    url = row["url"]
    start = row["start time"]
    end = row["end time"]
    print(f"video ID: {video_id}")
    print(f"Category: {category}")
    print(f"Caption: {caption}")
    
    # Temporary and final file paths
    temp_path = f"{save_dir}/{video_id}_full.mp4"
    clip_path = f"{save_dir}/{video_id}_{category}_{caption}.mp4"
    
    # Check if the clip already exists
    if os.path.exists(clip_path):
        print(f"Already exists: {clip_path}")
        continue

    # Download and trim the video
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "outtmpl": temp_path,
        "quiet": True,
    }

    try:
        print(f"Downloading: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Trim the video
        print(f"Trimming: {temp_path} -> {clip_path}")
        clip = VideoFileClip(temp_path).subclipped(start, end)
        clip.write_videofile(clip_path, codec="libx264", audio_codec="aac")
        clip.close()

        # Remove the temporary full video file
        os.remove(temp_path)

    except Exception as e:
        print(f"Error for {video_id}: {e}")
