import os
# Set environment variables for Hugging Face cache
os.environ["HF_HOME"] = "/scratch/rs02358/huggingface"
os.environ["TMPDIR"] = "/scratch/rs02358/tmp"

import cv2
import subprocess
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel

import shutil
RM_FRAME = True  # Set to True to remove frames after evaluation

# initialize models
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to("cuda").eval()

pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to("cuda")

def extract_frames(video_path: str, output_dir: str, fps: int = 1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        raise ValueError(f"Invalid FPS detected in video: {video_fps}")
    frame_interval = max(int(video_fps // fps), 1)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            filename = os.path.join(output_dir, f"frame_{saved:04d}.png")
            cv2.imwrite(filename, frame)
            saved += 1
        count += 1

    cap.release()

def calculate_clip_score_video(frame_dir, text):
    scores = []
    for filename in sorted(os.listdir(frame_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(os.path.join(frame_dir, filename)).convert("RGB")
            inputs = clip_processor(text=[text], images=[image], return_tensors="pt").to(clip_model.device)
            with torch.no_grad():
                # obtain image and text embeddings
                outputs = clip_model(**inputs)
                image_embed = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embed = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
                # calculate cosine similarity score
                score = torch.cosine_similarity(image_embed, text_embed).item()
                scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

def calculate_pickscore_video(frame_dir, text):
    scores = []
    for filename in sorted(os.listdir(frame_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(os.path.join(frame_dir, filename)).convert("RGB")

            # Obtain text embedding
            text_inputs = pick_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                text_emb = pick_model.get_text_features(**text_inputs)
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

            # Obtain image embedding
            image_inputs = pick_processor(images=[image], return_tensors="pt").to("cuda")
            with torch.no_grad():
                image_emb = pick_model.get_image_features(**image_inputs)
                image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)

            # Calculate PickScore
            score = (pick_model.logit_scale.exp() * (text_emb @ image_emb.T))[0][0].item()
            scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0

def evaluate_video(video_path, prompt, frame_dir="./frames", fps=1):
    extract_frames(video_path, frame_dir, fps=fps)
    clip_score = calculate_clip_score_video(frame_dir, prompt)
    pick_score = calculate_pickscore_video(frame_dir, prompt)
    if RM_FRAME:
        shutil.rmtree(frame_dir)
    return {"clip_score": clip_score, "pick_score": pick_score}

if __name__ == "__main__":
    result = evaluate_video("animation-0019_100steps.mp4", "a bear is walking, anime style")
    print(f"CLIP Score: {result['clip_score']:.4f}")
    print(f"PickScore : {result['pick_score']:.4f}")
