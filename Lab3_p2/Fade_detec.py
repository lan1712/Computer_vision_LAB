#Nhóm: 10
#MSSV: 054205005878
#Họ và tên: Phạm Vũ Lân
import os
import cv2 
import torch 
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1 
from scipy.spatial.distance import cosine

# --- CẤU HÌNH ---
DATA_FOLDER = "E:/SCHOOLS/Computer_Vision/data"  # thư mục chứa các ảnh mẫu
THRESHOLD = 0.7  # ngưỡng để coi là "Matched"
THUMB_SIZE = (160, 160)  # kích thước thumbnail hiển thị trên frame
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- BƯỚC 1: KHỞI TẠO MÔ HÌNH ---
print("Đang tải mô hình MTCNN + FaceNet (PyTorch)...")
detector = MTCNN(keep_all=True, device=device)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- BƯỚC 2: HÀM HỖ TRỢ ---
def get_embedding(face_img):
    # """Quy trình: BGR -> RGB -> Resize -> Tensor -> Normalize Embedding."""
    """Nhận ảnh BGR (OpenCV), trả về embedding chuẩn hóa (1D numpy)."""
    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160))
    img_tensor = torch.tensor(img_rgb/255., dtype=torch.float32).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embedder(img_tensor).cpu().numpy()[0]
    return emb / np.linalg.norm(emb)

def load_image(path):
    """Đọc ảnh bằng OpenCV, trả về None nếu không đọc được."""
    return cv2.imread(path)

def make_thumbnail(img, thumb_size=THUMB_SIZE):
    """Tạo thumbnail BGR giữ tỉ lệ (crop center square rồi resize)."""
    if img is None:
        return None
    h, w = img.shape[:2]
    min_side = min(h, w)
    cy, cx = h // 2, w // 2
    half = min_side // 2
    crop = img[cy-half:cy-half+min_side, cx-half:cx-half+min_side]
    thumb = cv2.resize(crop, thumb_size)
    return thumb

# --- BƯỚC 3: TẠO EMBEDDING CHO TOÀN BỘ ẢNH TRONG FOLDER data ---
print(f"Đang tải ảnh mẫu từ: {DATA_FOLDER}")
known_embeddings = []
known_names = []
known_thumbs = []  # lưu thumbnail để ghép lên frame

for fname in os.listdir(DATA_FOLDER):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    fullpath = os.path.join(DATA_FOLDER, fname)
    img = load_image(fullpath)
    if img is None:
        print(f"Không thể đọc ảnh: {fname}, bỏ qua.")
        continue

    boxes, _ = detector.detect(img)
    if boxes is None:
        print(f"Không tìm thấy khuôn mặt trong ảnh mẫu: {fname}, bỏ qua.")
        continue

    x1, y1, x2, y2 = map(int, boxes[0])
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    face_roi = img[y1:y2, x1:x2]
    if face_roi.size == 0:
        print(f"ROI rỗng cho ảnh: {fname}, bỏ qua.")
        continue

    emb = get_embedding(face_roi)
    known_embeddings.append(emb)
    known_names.append(fname)
    known_thumbs.append(make_thumbnail(img, THUMB_SIZE))
    print(f"Đã xử lý mẫu: {fname}")

if len(known_embeddings) == 0:
    print("Không có ảnh mẫu hợp lệ trong thư mục. Kết thúc.")
    exit()

known_embeddings = np.stack(known_embeddings)  # shape: (N, 128)
print(f"Đã tạo embedding cho {len(known_names)} ảnh mẫu. Bắt đầu bật Webcam...")

# --- BƯỚC 4: VÒNG LẶP WEBCAM VỚI HIỂN THỊ THUMBNAIL TRÊN FRAME ---

#Luồng xử lý lặp lại: Đọc khung hình -> Phát hiện mặt -> So sánh vector -> Hiển thị kết quả.
cap = cv2.VideoCapture(0)
last_printed = None  # tránh in lặp quá nhiều lần cùng 1 kết quả

while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes, _ = detector.detect(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # đảm bảo tọa độ hợp lệ trong khung hình
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue

            current_embedding = get_embedding(face_roi)

            # So sánh với tất cả embedding mẫu, tìm best match
            distances = np.array([cosine(current_embedding, ke) for ke in known_embeddings])
            similarities = 1 - distances
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            best_name = known_names[best_idx]
            best_thumb = known_thumbs[best_idx]

            # In ra tên file giống nhất và score (chỉ in khi khác so với lần in trước)
            info_str = f"Best match: {best_name} | Score: {best_score:.4f}"
            if info_str != last_printed:
                print(info_str)
                last_printed = info_str

            # Gán nhãn hiển thị trên khung hình
            if best_score >= THRESHOLD:
                label = f"{best_name} ({best_score:.2f})"
                color = (0, 255, 0)
            else:
                label = f"Unknown ({best_score:.2f})"
                color = (0, 0, 255)

            # Vẽ bounding box và nhãn
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # --- Ghép thumbnail lên góc phải trên của frame ---
            if best_thumb is not None:
                fh, fw = frame.shape[:2]
                th, tw = best_thumb.shape[:2]
                x_offset = fw - tw - 10  # 10 px cách mép phải
                y_offset = 10            # 10 px cách mép trên
                if x_offset >= 0 and y_offset + th <= fh:
                    # Ghi đè thumbnail lên frame
                    frame[y_offset:y_offset+th, x_offset:x_offset+tw] = best_thumb
                    # Vẽ viền nhỏ quanh thumbnail để nổi bật
                    cv2.rectangle(frame, (x_offset, y_offset), (x_offset+tw, y_offset+th), (255,255,255), 1)

    # Hiển thị frame chính (chỉ 1 cửa sổ)
    cv2.imshow("FaceNet Match-to-Folder Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
