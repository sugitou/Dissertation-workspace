import cv2

input_path = '/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_CleanAndJerk_g25_c01.avi'
output_path = '/scratch/rs02358/ved_dissertation/Datasets_from_Internet/UCF101/UCF-Benchmark/v_CleanAndJerk_g25_c01.mp4'

cap = cv2.VideoCapture(input_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()
