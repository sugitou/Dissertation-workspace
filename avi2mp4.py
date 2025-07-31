import cv2

input_path = '/scratch/rs02358/ved_dissertation/CCEdit/assets/ucf101/Inference/v_SkateBoarding_g01_c03.avi'
output_path = '/scratch/rs02358/ved_dissertation/CCEdit/assets/ucf101/Inference/v_SkateBoarding_g01_c03.mp4'

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
