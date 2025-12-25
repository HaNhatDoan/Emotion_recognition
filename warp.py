import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import os

# --- HÀM WARP ĐÃ SỬA LỖI (QUAN TRỌNG) ---
def warp_face_mp_corrected(img, src_lms_norm, mean_lms_norm, size=(48,48)):
    # Lấy kích thước thật của ảnh đầu vào (VD: 640x480)
    h_img, w_img = img.shape[:2]
    
    # 1. Tính tọa độ điểm nguồn trên ảnh gốc (Pixel thật)
    src_pts = src_lms_norm.reshape(-1, 2).copy()
    src_pts[:, 0] *= w_img # Nhân với chiều rộng thật
    src_pts[:, 1] *= h_img # Nhân với chiều cao thật
    
    # 2. Tính tọa độ điểm đích trên ảnh 48x48 (Mean Shape)
    dst_pts = mean_lms_norm.reshape(-1, 2) * size[0]

    if len(img.shape) == 3: 
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else: 
        img_gray = img
        
    # 3. Tính ma trận biến đổi
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    
    if M is not None:
        warped = cv2.warpAffine(img_gray, M, size)
    else:
        warped = cv2.resize(img_gray, size)
    
    return warped

# --- MAIN ---
# 1. Load Mean Shape
if not os.path.exists("model_mp_main_tot_nhat/mean_shape.pkl"):
    print("Cần train model trước để có file mean_shape.pkl")
    exit()

mean_shape = joblib.load("model_mp_main_tot_nhat/mean_shape.pkl")
mp_face_mesh = mp.solutions.face_mesh

# 2. Mở Webcam
cap = cv2.VideoCapture(0)
print("Hãy NGHIÊNG ĐẦU sang một bên và nhấn SPACE để chụp...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Lật ảnh cho giống gương
    frame = cv2.flip(frame, 1)
    
    cv2.imshow("Chup anh minh hoa (Nhan SPACE)", frame)
    if cv2.waitKey(1) == 32: # Space
        original_img = frame
        break
cap.release()
cv2.destroyAllWindows()

# 3. Xử lý
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    
    if res.multi_face_landmarks:
        # Lấy landmark
        lm_list = []
        for i, lm in enumerate(res.multi_face_landmarks[0].landmark):
            if i >= 468: break
            lm_list.extend([lm.x, lm.y])
            
        lm_raw = np.array(lm_list, dtype=np.float32)
        
        # --- GỌI HÀM WARP ĐÃ SỬA ---
        warped_img = warp_face_mp_corrected(original_img, lm_raw, mean_shape, (48, 48))

        # 4. Vẽ biểu đồ so sánh cho Báo cáo
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        # Ảnh 1: Gốc (Vẽ thêm vài landmark cho đẹp)
        vis_img = original_img.copy()
        h, w, _ = vis_img.shape
        for i in range(0, 936, 20): # Vẽ thưa thưa thôi
            cx, cy = int(lm_raw[i]*w), int(lm_raw[i+1]*h)
            cv2.circle(vis_img, (cx, cy), 2, (0, 255, 0), -1)
            
        ax[0].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        ax[0].set_title("1. Ảnh gốc (Đầu nghiêng)")
        ax[0].axis('off')
        
        # Ảnh 2: Kết quả (Thẳng)
        ax[1].imshow(warped_img, cmap='gray')
        ax[1].set_title("2. Ảnh sau khi Warp (48x48)")
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("Không tìm thấy mặt trong ảnh vừa chụp!")