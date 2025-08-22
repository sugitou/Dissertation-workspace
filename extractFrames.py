import cv2
import numpy as np
import os
import glob

def extract_uniform_frames(video_path, outdir, num_frames=8, prefix="frame"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0:
        raise RuntimeError("Cannot read frame count")

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)

    os.makedirs(outdir, exist_ok=True)
    saved = 0
    for i, idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        time_sec = idx / fps if fps and fps > 0 else 0
        outname = f"{prefix}_{i:03d}_f{idx}_t{time_sec:.2f}s.png"
        cv2.imwrite(os.path.join(outdir, outname), frame)
        saved += 1

    cap.release()
    print(f"Saved {saved} frames to {outdir}")

def make_horizontal_strip(
    images_or_glob,
    out_path,
    target_height=320,
    pad=8,
    pad_color=(255, 255, 255),
    draw_filenames=False,
    font_scale=0.5,
    font_thickness=1
):
    """
    images_or_glob: 画像パスのリスト もしくは glob パターン（例: 'frames_out/*.png'）
    out_path: 出力PNGのパス
    target_height: ストリップ内での各画像の高さ（等倍リサイズ）
    pad: 画像間と外枠の余白（ピクセル）
    pad_color: 余白の色（BGR）
    draw_filenames: 画像下にファイル名ラベルを描画（True/False）
    """
    # 画像パスの解決
    if isinstance(images_or_glob, str):
        img_paths = sorted(glob.glob(images_or_glob))
    else:
        img_paths = list(images_or_glob)

    if len(img_paths) == 0:
        raise ValueError("No images found")

    # 画像を読み込み＆高さを揃えて横連結
    resized_imgs = []
    labels = []
    for p in img_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip unreadable: {p}")
            continue
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = max(1, int(round(w * scale)))
        resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
        resized_imgs.append(resized)
        labels.append(os.path.basename(p))

    if len(resized_imgs) == 0:
        raise ValueError("All images failed to load")

    # ラベル領域の高さ（任意）
    label_h = 0
    if draw_filenames:
        # おおよそ文字の高さを見積もって確保
        label_h = int(20 + 10 * font_scale)

    # キャンバス幅・高さを計算（外枠＆間隔の余白込み）
    widths = [im.shape[1] for im in resized_imgs]
    total_w = sum(widths) + pad * (len(resized_imgs) - 1) + pad * 2
    total_h = target_height + label_h + pad * 2

    canvas = np.full((total_h, total_w, 3), pad_color, dtype=np.uint8)

    x = pad
    y_img_top = pad
    y_label_top = pad + target_height + 5  # 5pxのマージン

    for im, name in zip(resized_imgs, labels):
        h, w = im.shape[:2]
        canvas[y_img_top:y_img_top + h, x:x + w] = im

        if draw_filenames:
            # 影付きで見やすく
            org = (x + 5, y_label_top + int(12 * font_scale))
            cv2.putText(canvas, name, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, name, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (30, 30, 30), font_thickness, cv2.LINE_AA)

        x += w + pad

    # 保存
    ok = cv2.imwrite(out_path, canvas)
    if not ok:
        raise RuntimeError(f"Failed to save: {out_path}")
    print(f"Saved strip: {out_path}")

# ===== 使い方例 =====
if __name__ == "__main__":
    # 1) 等間隔抽出
    extract_uniform_frames(
        "/scratch/rs02358/ved_dissertation/CCEdit/outputs/tv2v/Thinking/CCEdit-20250821T143656Z-1-001/CCEdit/Boxing/15s.mp4",
        # "/scratch/rs02358/ved_dissertation/MotionDirector/outputs/Thinking/A_crystal_skeleton_is_brushing_its_teeth_in_a_bathroom_785490_0.3.mp4",
        # "/scratch/rs02358/ved_dissertation/MotionDirector/outputs/Thinking/A_focused_woman_carefully_applies_face_cream_with_gentle_motions_in_her_bathroom_785490_0.3.mp4",
        # "/scratch/rs02358/ved_dissertation/MotionDirector/outputs/Thinking/A_man_is_practising_his_golf_swing_in_Times_Square_New_York_785490_0.5.mp4",
        # "/scratch/rs02358/ved_dissertation/MotionDirector/outputs/Thinking/A_man_wearing_white_tank_top_practices_cooking_flipping_pancakes_in_his_garage_home_gym_785490_0.1.mp4",
        # "/scratch/rs02358/ved_dissertation/MotionDirector/outputs/Thinking/A_muscular_alien_performs_an_impressive_snatch_lifting_a_heavy_barbell_overhead_in_a_gym_785490_0.3.mp4",
        # "/scratch/rs02358/ved_dissertation/MotionDirector/outputs/Thinking/A_woman_in_a_maid's_costume_ascends_a_challenging_indoor_climbing_wall_with_a_follow_cinematic_shot_785490_0.1.mp4",
        # "/scratch/rs02358/ved_dissertation/MotionDirector/outputs/Thinking/An_athletic_person_holds_a_steady_handstand_on_a_sandy_beach_under_a_few_white_cloud_sky_785490_0.3.mp4",
        # "/scratch/rs02358/ved_dissertation/MotionDirector/outputs/Thinking/Two_focused_fencers_in_white_uniforms_duel_on_a_strip_inside_a_business_office_785490.mp4",
        "frames_out/Boxing/results-long",
        num_frames=8,
        prefix="frame")

    # 2) 横一列ストリップ生成（ファイル名ラベルなし）
    make_horizontal_strip("frames_out/Boxing/results-long/frame_*.png", "frames_out/Boxing/results-long/Seq_Boxing.png", target_height=120, pad=0, draw_filenames=False)

    # 3) ラベル付きで作る場合
    # make_horizontal_strip("frames_out/*.png", "strip_labeled.png", target_height=360, pad=12, draw_filenames=True)
