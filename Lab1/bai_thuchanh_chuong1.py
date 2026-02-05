import os
import cv2
import numpy as np
from PIL import Image
 
IMG_PATH = "test.png"      
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

''' def show(title, img, wait=True, max_w=1200, max_h=700):
    # Thu nhỏ ảnh để vừa màn hình (không phóng to)
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Cho phép resize cửa sổ nếu muốn kéo tay
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    if wait:
        cv2.waitKey(0)
'''
def show(title, img, wait=True):
    
    cv2.imshow(title, img)
    if wait:
        cv2.waitKey(0)
def main():

    img_bgr = cv2.imread(IMG_PATH)   
    if img_bgr is None:
        raise FileNotFoundError(f"Không đọc được ảnh. Kiểm tra lại IMG_PATH = {IMG_PATH}")

    print("Kích thước (H, W, C):", img_bgr.shape)

    show("Original (BGR)", img_bgr)

    # Lưu lại sang định dạng khác (vd: PNG)
    out_jpg = os.path.join(OUT_DIR, "01_original.jpg")
    cv2.imwrite(out_jpg, img_bgr)
    print("Đã lưu:", out_jpg)

    # (Pillow) Đọc và lưu lại (vd: WEBP) để đúng yêu cầu “OpenCV và Pillow”
    pil_img = Image.open(IMG_PATH)
    out_webp = os.path.join(OUT_DIR, "01_original.webp")
    pil_img.save(out_webp, "WEBP", quality=90)
    print("Đã lưu:", out_webp)
 
    # 3) CHUYỂN ĐỔI KHÔNG GIAN MÀU
 
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out_gray = os.path.join(OUT_DIR, "02_gray.png")
    cv2.imwrite(out_gray, gray)
    print("Đã lưu:", out_gray)
    show("Grayscale", gray)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    out_hsv = os.path.join(OUT_DIR, "03_hsv.png")
    cv2.imwrite(out_hsv, hsv)  
    print("Đã lưu (HSV raw):", out_hsv)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    out_lab = os.path.join(OUT_DIR, "04_lab.png")
    cv2.imwrite(out_lab, lab)  
    print("Đã lưu (LAB raw):", out_lab)
    """ 
     h, s, v = cv2.split(hsv)
    cv2.imwrite(os.path.join(OUT_DIR, "03_h.png"), h)
    cv2.imwrite(os.path.join(OUT_DIR, "03_s.png"), s)
    cv2.imwrite(os.path.join(OUT_DIR, "03_v.png"), v)

    l, a, b = cv2.split(lab)
    cv2.imwrite(os.path.join(OUT_DIR, "04_l.png"), l)
    cv2.imwrite(os.path.join(OUT_DIR, "04_a.png"), a)
    cv2.imwrite(os.path.join(OUT_DIR, "04_b.png"), b)"""
     
    # 4) CẮT XÉN & THAY ĐỔI KÍCH THƯỚC
 
    H, W = img_bgr.shape[:2]

    # Cắt 1 vùng bất kỳ (ví dụ: cắt giữa ảnh)
    x1, y1 = int(W * 0.25), int(H * 0.25)
    x2, y2 = int(W * 0.75), int(H * 0.75)
    crop = img_bgr[y1:y2, x1:x2]
    out_crop = os.path.join(OUT_DIR, "05_crop.png")
    cv2.imwrite(out_crop, crop)
    print("Đã lưu:", out_crop)
    show("Crop", crop)

    # Resize theo tỷ lệ (ví dụ: 50%)
    scale = 0.5
    resized_ratio = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out_resize_ratio = os.path.join(OUT_DIR, "06_resize_ratio_50.png")
    cv2.imwrite(out_resize_ratio, resized_ratio)
    print("Đã lưu:", out_resize_ratio)
    show("Resize 50%", resized_ratio)

    # Resize kích thước cố định (ví dụ: 640x360)
    fixed_w, fixed_h = 640, 360
    resized_fixed = cv2.resize(img_bgr, (fixed_w, fixed_h), interpolation=cv2.INTER_AREA)
    out_resize_fixed = os.path.join(OUT_DIR, "07_resize_640x360.png")
    cv2.imwrite(out_resize_fixed, resized_fixed)
    print("Đã lưu:", out_resize_fixed)
    show("Resize 640x360", resized_fixed)

    
    # 5) VẼ HÌNH CƠ BẢN + THÊM VĂN BẢN
   
    canvas = img_bgr.copy()

    cv2.line(canvas, (20, 20), (W - 20, H - 20), (0, 255, 0), 3)

    cv2.rectangle(canvas, (50, 50), (250, 180), (255, 0, 0), 3)

    cv2.circle(canvas, (W // 2, H // 2), 80, (0, 0, 255), 3)
    cv2.putText(
        canvas,
        "Xu ly anh & Thi giac may tinh",
        (50, H - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    out_draw = os.path.join(OUT_DIR, "08_draw_shapes_text.png")
    cv2.imwrite(out_draw, canvas)
    print("Đã lưu:", out_draw)
    show("Draw + Text", canvas)

    cv2.destroyAllWindows()
    print("\nXONG! Tat ca output nam trong thu muc:", OUT_DIR)

if __name__ == "__main__":
    main()
