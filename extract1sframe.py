import cv2
import os

# 入力動画のパスをリストに格納
video_list = [
    "/scratch/rs02358/ved_dissertation/CCEdit/outputs/tv2v/HandstandWalking/revAnimated_v2Rebirth/result/mp4/animation-0001.mp4",
    "/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_HandstandWalking_g14_c03.mp4",
]

# 保存先フォルダ
output_dir = "frames_1s"
os.makedirs(output_dir, exist_ok=True)

for video_path in video_list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: {video_path} を開けませんでした")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)  # FPSを取得
    frame_number = int(fps * 1)      # 1秒目のフレーム番号
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_1s.png")
        cv2.imwrite(out_path, frame)
        print(f"保存しました: {out_path}")
    else:
        print(f"Error: {video_path} の1秒目のフレームを取得できませんでした")
    
    cap.release()
