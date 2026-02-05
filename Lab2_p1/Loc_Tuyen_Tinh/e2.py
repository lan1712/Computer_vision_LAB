# các yêu cầu từ 1 đến 3 đều làm trong file này
import cv2
import numpy as np

img = cv2.imread("pikachu.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original", img)


# 1. Lọc trung bình: Làm mờ ảnh bằng cách tính giá trị trung bình của các điểm ảnh trong một vùng lân cận.
blur = cv2.blur(gray, (5, 5))
cv2.imshow("1. Blur", blur)




# 2. Lọc Gaussian: Làm mờ ảnh một cách tự nhiên hơn bằng cách sử dụng kernel Gaussian.
gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("2. Gaussian Blur", gaussian_blur)


# 3. Làm sắc nét: Tăng cường các cạnh trong ảnh bằng cách sử dụng các kernel thích hợp.
kernel_shappen = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
sharpen = cv2.filter2D(gray, -1, kernel_shappen)
cv2.imshow("3. Sharpen", sharpen)

cv2.waitKey(0)
cv2.destroyAllWindows()