import cv2
import numpy as np
import matplotlib.pyplot as plt
from sift import des1, des2, kp1, kp2, img1, img2
# Initialize Brute-Force Matcher and find 2 nearest neighbors as given in the instructions
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

ratios = []
match_data = [] # This will store the ratio, the first match object, and the second match object for the top 5 matches.
nndr_threshold = 0.8 

for m, n in matches:
    ratio = m.distance / n.distance
    ratios.append(ratio)
    match_data.append((ratio, m, n))

# Filter for the full correspondence visualization
good_matches = [m for ratio, m, n in match_data if ratio < nndr_threshold]

# Sort to find the top 5 distinct matches
match_data.sort(key=lambda x: x[0])
top_5_data = match_data[:5]

plt.figure(figsize=(10, 6))
plt.hist(ratios, bins=50, color='skyblue', edgecolor='black')
plt.axvline(nndr_threshold, color='red', linestyle='dashed', linewidth=2, label=f'Threshold: {nndr_threshold}')
plt.title('Histogram of NNDR Ratios')
plt.xlabel('Ratio (d1 / d2)')
plt.ylabel('Frequency')
plt.legend()
plt.show()

def extract_patch(img, kp, patch_size=41):
    x, y = int(kp.pt[0]), int(kp.pt[1])
    r = patch_size // 2
    padded = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REPLICATE)
    return padded[y:y+patch_size, x:x+patch_size]

fig, axes = plt.subplots(5, 3, figsize=(12, 18))
plt.suptitle('Top 5 Best Feature Matches by NNDR (L2) - RGB', fontsize=16)

cols = ['Image 1 Feature', 'Nearest Neighbor (1st)', 'Second Nearest (2nd)']
for ax, col in zip(axes[0], cols):
    ax.set_title(col, fontweight='bold')

for i, (ratio, m, n) in enumerate(top_5_data):
    # Extracting patches as Query (Img1), 1st NN (Img2), 2nd NN (Img2)
    p1 = extract_patch(img1, kp1[m.queryIdx])
    p2 = extract_patch(img2, kp2[m.trainIdx])
    p3 = extract_patch(img2, kp2[n.trainIdx])
    
    # Plotting
    for j, patch in enumerate([p1, p2, p3]):
        axes[i, j].imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        axes[i, j].axis('off')
        
    # Add NNDR label to the first column
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[i, 0].text(0.05, 0.95, f'NNDR={ratio:.3f}', transform=axes[i, 0].transAxes, 
                    fontsize=9, verticalalignment='top', bbox=props)
    axes[i, 0].set_ylabel(f"Match #{i+1}", rotation=0, labelpad=40, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(20, 10))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f'Correspondences (NNDR < {nndr_threshold})')
plt.axis('off')
plt.show()