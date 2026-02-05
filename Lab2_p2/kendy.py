import cv2
import numpy as np
from PIL import Image
import sys
import os

image_path = 'image.jpg'

# Kiểm tra file tồn tại
if not os.path.exists(image_path):
	print(f"File not found: {image_path}")
	sys.exit(1)

# Đọc ảnh bằng OpenCV
image = cv2.imread(image_path)
if image is None:
	print(f"Không thể đọc ảnh bằng OpenCV: {image_path}")
	sys.exit(1)

# Thay đổi kích thước ảnh
resized_image = cv2.resize(image, (200, 100))

# Xoay ảnh 45 độ (sửa tên biến cols)
rows, cols = image.shape[:2]
center = (cols / 2.0, rows / 2.0)
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image, M, (cols, rows))

# Hiển thị ảnh gốc và ảnh đã chỉnh sửa
cv2.imshow('Original Image', image)
cv2.imshow('Resized Image', resized_image)
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Cũng minh họa bằng PIL (tuỳ chọn)
try:
	pil_image = Image.open(image_path)
except Exception as e:
	print(f"Không thể đọc ảnh bằng PIL: {e}")
else:
	pil_resized = pil_image.resize((200, 100))
	pil_rotated = pil_image.rotate(45, expand=True)
	pil_resized.show()
	pil_rotated.show()