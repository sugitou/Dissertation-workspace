import os
# Set environment variables for Hugging Face cache
os.environ["HF_HOME"] = "/scratch/rs02358/huggingface"
os.environ["TMPDIR"] = "/scratch/rs02358/tmp"

import re
import cv2
import shutil
import subprocess
import tempfile
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel, AutoProcessor, AutoModel


# initialize models
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to("cuda").eval()

pick_processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
pick_model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to("cuda")


# Convert video codec to a browser-compatible MP4 format
# Unstable, that's why don't use it in displayEval.py 
def convert_codec(input_path):
    if not input_path.endswith(".mp4"):
        return input_path  # no conversion needed

    # Use ffmpeg to convert the video
    output_path = tempfile.mktemp(suffix=".mp4")

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",        # video codec: H.264
        "-preset", "ultrafast",   # preset for fast encoding
        "-c:a", "aac",            # audio codec: AAC
        "-movflags", "+faststart",  # for better streaming
        output_path
    ]

    try:
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise RuntimeError(f"FFmpeg created invalid file: {output_path}")
        return output_path
    except Exception as e:
        print(f"[FFMPEG ERROR] {e}")
        return input_path  # fallback


# Extract information about the video
def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Could not open video", "", "", "", ""
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = f"{width}x{height}"

    cap.release()
    return os.path.basename(video_path), f"{duration:.2f} sec", str(frame_count), f"{fps:.2f}", resolution


def extract_frames(video_path: str, output_dir: str, fps: int = 1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        raise ValueError(f"Invalid FPS detected in video: {video_fps}")
    frame_interval = max(int(video_fps // fps), 1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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

# Calculate CLIP similarity score for video frames with a given text prompt 
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


# Calculate PickScore for video frames with a given text prompt
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


def natural_key(s: str):
    # 数字部分を整数化して分割、例: frame_2 < frame_10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


# Calculate temporal consistency score for video frames between Frame t and Frame t+1
def calculate_temporal_consistency(frame_dir):
    embeddings = []
    files = [f for f in os.listdir(frame_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort(key=natural_key)

    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(os.path.join(frame_dir, filename)).convert("RGB")
            inputs = clip_processor(images=[image], return_tensors="pt").to("cuda")
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                embeddings.append(emb)

    similarities = [
        torch.cosine_similarity(embeddings[i], embeddings[i + 1]).item()
        for i in range(len(embeddings) - 1)
    ]
    return sum(similarities) / len(similarities) if similarities else 0.0


def calculate_temporal_consistency_lag(frame_dir, k=1):
    """
    Temporal consistency score using CLIP embeddings with lag-k frames.
    
    Args:
        frame_dir (str): フレーム画像のディレクトリ
        clip_model: CLIP のモデル
        clip_processor: CLIP の前処理
        device (str): "cuda" または "cpu"
        k (int): ラグ幅（例: 1=隣接, 2=2フレーム間隔,...）
    
    Returns:
        float: ラグ k の temporal consistency スコア
    """
    # --- フレーム埋め込み ---
    embeddings = []
    files = [f for f in os.listdir(frame_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort(key=natural_key)

    for filename in files:
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(os.path.join(frame_dir, filename)).convert("RGB")
            inputs = clip_processor(images=[image], return_tensors="pt").to("cuda")
            with torch.no_grad():
                emb = clip_model.get_image_features(**inputs)
                emb = emb / emb.norm(p=2, dim=-1, keepdim=True)  # L2正規化
                embeddings.append(emb)
    
    if len(embeddings) <= k:
        return 0.0  # フレーム数が足りない場合
    
    # --- 類似度計算 ---
    similarities = [
        torch.cosine_similarity(embeddings[i], embeddings[i + k]).item()
        for i in range(len(embeddings) - k)
    ]
    
    return sum(similarities) / len(similarities) if similarities else 0.0


# Calculate appearance diversity score for video frames with a list of prompts
def calculate_appearance_diversity(frame_dir, prompt_list):
    scores = []
    for prompt in prompt_list:
        text_inputs = clip_processor(text=[prompt], return_tensors="pt").to("cuda")
        with torch.no_grad():
            # Obtain text embedding
            text_emb = clip_model.get_text_features(**text_inputs)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

        for filename in sorted(os.listdir(frame_dir)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                image = Image.open(os.path.join(frame_dir, filename)).convert("RGB")
                inputs = clip_processor(images=[image], return_tensors="pt").to("cuda")
                with torch.no_grad():
                    # Obtain image embedding
                    image_emb = clip_model.get_image_features(**inputs)
                    image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
                
                # calculate cosine similarity score
                score = torch.cosine_similarity(text_emb, image_emb).item()
                scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0



def calculate_embedding_distance(frame_dir, prompt):
    # Obtain average image embedding from all frames
    image_embeds = []
    for filename in sorted(os.listdir(frame_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(os.path.join(frame_dir, filename)).convert("RGB")
            inputs = clip_processor(images=[image], return_tensors="pt").to(clip_model.device)
            with torch.no_grad():
                image_emb = clip_model.get_image_features(**inputs)
                image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
                image_embeds.append(image_emb.cpu().numpy().flatten())
    if not image_embeds:
        return None
    avg_image_emb = np.mean(image_embeds, axis=0)

    # Obtain text embedding for the prompt
    text_inputs = clip_processor(text=[prompt], return_tensors="pt").to(clip_model.device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
    text_emb = text_emb.cpu().numpy().flatten()

    # Calculate cosine similarity and distance
    sim = np.dot(avg_image_emb, text_emb) / (np.linalg.norm(avg_image_emb) * np.linalg.norm(text_emb))
    dist = 1 - sim  # cosine distance
    return dist


def evaluate_video(video_path, prompt, frame_dir="./frames", fps=8):
    # Create a temporary directory for frames
    frame_dir = tempfile.mkdtemp(prefix="frames_")
    lagk_tc_list = []
    
    try:
        extract_frames(video_path, frame_dir, fps=fps)
        clip_score = calculate_clip_score_video(frame_dir, prompt)
        pick_score = calculate_pickscore_video(frame_dir, prompt)
        # temporal_consistency = calculate_temporal_consistency(frame_dir)
        embedding_distance = calculate_embedding_distance(frame_dir, prompt)
        for k in [1, 4, 8, 12, 16, 20, 24]:
            lagk_tc = calculate_temporal_consistency_lag(frame_dir, k=k)
            lagk_tc_list.append(lagk_tc)

        return {
            "clip_score": clip_score, 
            "pick_score": pick_score, 
            "Tem-Con": lagk_tc_list, 
            "embedding_distance": embedding_distance,
        }

    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


if __name__ == "__main__":
    prompt_list = [
        "a bear is walking, anime style",
        "an animal walking in nature",
        "a cartoon character in a landscape",
        "a person running in a city",
        "a bird flying in the sky",
    ]
    
    # result = evaluate_video("/scratch/rs02358/ved_dissertation/CCEdit/outputs/tv2v/Thinking/Boxing-pixel_prior0.3_cfg9.mp4", "A man wearing white tank top practices boxing, punching a red heavy bag in his garage home gym, pixel art style")
    result = evaluate_video("/scratch/rs02358/ved_dissertation/Datasets_from_Internet/Boxing-pixel_10s.mp4", "A man wearing white tank top practices boxing, punching a red heavy bag in his garage home gym, pixel art style")
    print(f"CLIP Score: {result['clip_score']:.4f}")
    print(f"PickScore : {result['pick_score']:.4f}")
    # print(f"Temporal Consistency: {result['Tem-Con']:.4f}")
    print(f"Lag-k Temporal Consistency: {[round(tc, 4) for tc in result['Tem-Con']]}")
    print(f"Embedding Distance: {result['embedding_distance']:.4f}")
    # print(f"Lag-k Temporal Consistency: {[round(tc, 4) for tc in result['lagk_tc']]}")
    # print(f"Lag-k Temporal Consistency: {result['lagk_tc']}")