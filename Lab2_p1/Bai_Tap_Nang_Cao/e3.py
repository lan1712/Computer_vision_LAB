import cv2
import numpy as np

# convert image to gray image
img = cv2.imread("pikachu.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1. Phát hiện cạnh: Sử dụng các kernel Sobel, Prewitt để phát hiện các cạnh trong ảnh

# --- 1. Sobel ---
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel_mag = cv2.magnitude(sobel_x, sobel_y)
sobel = cv2.convertScaleAbs(sobel_mag)

# --- 2. Prewitt (Kernel tự thiết kế) ---
prewitt_x = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

prewitt_y = np.array([
    [ 1,  1,  1],
    [ 0,  0,  0],
    [-1, -1, -1]
], dtype=np.float32)

px = cv2.filter2D(gray, cv2.CV_64F, prewitt_x)
py = cv2.filter2D(gray, cv2.CV_64F, prewitt_y)

prewitt = cv2.convertScaleAbs(np.abs(px) + np.abs(py))  

# 2. KERNEL TÙY CHỈNH – HIỆU ỨNG CHẠM NỔI (EMBOSS)

kernel_emboss = np.array([
    [-2, -1,  0],
    [-1,  1,  1],
    [ 0,  1,  2]
], dtype=np.float32)

emboss = cv2.filter2D(gray, -1, kernel_emboss)

#3. So sánh các loại lọc: So sánh hiệu quả của các loại lọc khác nhau trên cùng một hình ảnh

# Median Filter – khử nhiễu muối tiêu
median_img = cv2.medianBlur(gray, 5)

# Bilateral Filter – làm mịn nhưng giữ biên
bilateral_img = cv2.bilateralFilter(gray, 9, 75, 75)
#4. Đối chứng với lọc Gaussian

# Gaussian Filter – lọc tuyến tính (đối chứng)
gaussian_img = cv2.GaussianBlur(gray, (5, 5), 0)

# =====================================================
def resize(img, size=(300, 300)):
    return cv2.resize(img, size)

img_cau1 = np.hstack((
    resize(gray),
    resize(sobel),
    resize(prewitt)
))
cv2.imshow("Cau 1 - Edge Detection (Original | Sobel | Prewitt)", img_cau1)

img_cau2 = np.hstack((
    resize(gray),
    resize(emboss)
))
cv2.imshow("Cau 2 - Custom Kernel (Original | Emboss)", img_cau2)



img_cau34 = np.hstack((
    resize(gaussian_img),
    resize(median_img),
    resize(bilateral_img)
))
cv2.imshow("Cau 3 & 4 - Filters Comparison (Gaussian | Median | Bilateral)", img_cau34)




# cv2.imshow("Original Gray", gray)

# #1.
# cv2.imshow("Sobel Edge", sobel)
# cv2.imshow("Prewitt Edge (Custom Kernel)", prewitt)
# #2.
# cv2.imshow("Emboss Filter", emboss)
# #3+4.
# cv2.imshow("Median Filter", median_img)
# cv2.imshow("Bilateral Filter", bilateral_img)
# cv2.imshow("Gaussian Filter", gaussian_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
