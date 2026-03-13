import argparse
import matplotlib.pyplot as plt
import cv2
import os
import torch
from colorizers import *

# --- CẤU HÌNH ---
# Đường dẫn thư mục chứa các ảnh đen trắng (thay bằng đường dẫn thật của bạn)
input_dir = r'D:\CPV\PE\colored_org'
# Đường dẫn thư mục lưu ảnh kết quả
save_dir = r'D:\CPV\PE\colored_keys'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# --- KHỞI TẠO MODEL ---
use_gpu = torch.cuda.is_available()  # Tự động kiểm tra GPU
colorizer_siggraph17 = siggraph17(pretrained=True).eval()

if use_gpu:
    colorizer_siggraph17.cuda()

# --- LẶP QUA TẤT CẢ ẢNH TRONG THƯ MỤC ---
# Lấy danh sách file và sắp xếp theo tên
files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
files.sort()

print(f"Tìm thấy {len(files)} ảnh. Đang bắt đầu xử lý...")

for filename in files:
    # 1. Đường dẫn file
    img_path = os.path.join(input_dir, filename)

    # 2. Load và Tiền xử lý
    img = load_img(img_path)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))

    if use_gpu:
        tens_l_rs = tens_l_rs.cuda()

    # 3. Colorize (Sử dụng Siggraph17)
    with torch.no_grad():  # Giúp tiết kiệm bộ nhớ khi inference
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # 4. Chuyển đổi và Lưu
    # Đổi tên từ frame_0239.jpg thành key_0239.png
    new_filename = filename.replace('frame_', 'key_').rsplit('.', 1)[0] + '.png'
    full_path = os.path.join(save_dir, new_filename)

    # Chuyển RGB -> BGR và ép kiểu uint8 để cv2 lưu đúng
    out_img_siggraph17_bgr = cv2.cvtColor((out_img_siggraph17 * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite(full_path, out_img_siggraph17_bgr)

    print(f"Xong: {new_filename}")

print(f"\n--- HOÀN THÀNH ---")
print(f"Toàn bộ ảnh đã được lưu tại: {save_dir}")