#các yêu cầu từ 1 đến 4 đều làm trong file này
import cv2
import numpy as np

#read img
img = cv2.imread('pikachu.jpg')
# cv2.imshow('Original Image', img)


#1. Thay đổi độ sáng: Tăng hoặc giảm độ sáng của toàn bộ hình ảnh bằng cách cộng hoặc trừ một giá trị cố định cho mỗi điểm ảnh.
brightness = 50  # Giá trị độ sáng 
bright_img = cv2.add(img, brightness )  # Tăng độ sáng
dim = -200
dim_img = cv2.add(img, dim )  # Giảm độ sáng

# cv2.imshow('Brightened Image', bright_img)

#2. Thay đổi độ tương phản: Tăng hoặc giảm độ tương phản bằng cách nhân mỗi điểm ảnh với một hằng số.
contrast = 1.5  # Hệ số tương phản
contrast_img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
# cv2.imshow('Contrasted Image', contrast_img)

#3. Biến đổi âm bản: Đảo ngược các giá trị điểm ảnh, tạo ra ảnh âm bản.
negative_img = 255 - img
# cv2.imshow('Negative Image', negative_img)

# 4. Cắt ngưỡng: Tạo ảnh nhị phân bằng cách so sánh giá trị mỗi điểm ảnh với một ngưỡng nhất định

threshold_gray = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)[1]
threshold_img = cv2.cvtColor(threshold_gray, cv2.COLOR_GRAY2BGR)

# cv2.imshow('Threshold Image', threshold_img)

def resize(img, size=(300, 300)):
    return cv2.resize(img, size)
img_cau1 = np.hstack((
    resize(img),
    resize(bright_img),
    resize(dim_img)
))
cv2.imshow("1. Origin | Brightness | dim ", img_cau1)

img_cau2 = np.hstack((
    resize(img),
    resize(contrast_img)
))
cv2.imshow("2. Origin | Contrast", img_cau2)

img_cau3 = np.hstack((
    resize(img),
    resize(negative_img)
))
cv2.imshow("3. Origin | Negative", img_cau3)

img_cau4 = np.hstack((
    resize(img),
    resize(threshold_img)
))
cv2.imshow("4. Origin | Threshold", img_cau4)

cv2.waitKey(0)
cv2.destroyAllWindows()