import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('../assets/staff_images/img1.jpeg')
img2 = cv2.imread('../assets/staff_images/img2.jpeg')

# Convert to grayscale as SIFT requires grayscale images
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect SIFT features
sift = cv2.SIFT_create()
# These are the parameters of the SIFT detector that I used. Fortunately, the default values are good enough for us. 
params = {
    "nfeatures": sift.getNFeatures(),
    "nOctaveLayers": sift.getNOctaveLayers(),
    "contrastThreshold": sift.getContrastThreshold(),
    "edgeThreshold": sift.getEdgeThreshold(),
    "sigma": sift.getSigma()
}

for param, value in params.items():
    print(f"{param}: {value}")

'''
The result from the SIFT detector:
    nfeatures: 0
    nOctaveLayers: 3
    contrastThreshold: 0.04
    edgeThreshold: 10.0
    sigma: 1.6

The total number of features for img1: 8164
The total number of features for img2: 7591
'''
#Finding keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# Drawing SIFT features on the images. Note that the flags is used to draw 'rich' keypoints and by rich, it gives us more information about the rotation and scale of the keypoints.
img1_sift = cv2.drawKeypoints(img1, kp1, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_sift = cv2.drawKeypoints(img2, kp2, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1_sift, cv2.COLOR_BGR2RGB))
plt.title(f'Image 1: {len(kp1)} features')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2_sift, cv2.COLOR_BGR2RGB))
plt.title(f'Image 2: {len(kp2)} features')
plt.show()