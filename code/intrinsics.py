import numpy as np
import cv2
# As both images are taken with the same camera, we can use the same intrinsic matrix for both images. Intrinsics is not going to change for the same camera. 
img1 = cv2.imread('../assets/staff_images/img1.jpeg')
h_px,w_px = img1.shape[:2]
f_mm = 6.765
w_mm = 9.757
h_mm = 7.318

fx = (f_mm * w_px) / w_mm
fy = (f_mm * h_px) / h_mm
cx = w_px / 2
cy = h_px / 2

K = np.array([[ fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
             ])
print(K)