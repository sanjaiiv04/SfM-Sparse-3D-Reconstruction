import numpy as np
import cv2
import matplotlib.pyplot as plt
from nndr import good_matches
from intrinsics import K
from sift import des1, des2, kp1, kp2, img1, img2

def compute_sampson_distance(pts1, pts2, E):
    '''
    This function computes the Sampson distance between two points and an essential matrix. Here is the reference:
    https://amroamroamro.github.io/mexopencv/matlab/cv.sampsonDistance.html
    '''
    # Converting to homogeneous coordinates
    p1 = np.column_stack((pts1, np.ones(len(pts1))))
    p2 = np.column_stack((pts2, np.ones(len(pts2))))
    
    # Numerator: (x2' E x1)^2
    Ep1 = (E @ p1.T).T
    p2E = (p2 @ E)
    numerator = np.sum(p2 * Ep1, axis=1)**2
    
    # Denominator: (Ex1)_1^2 + (Ex1)_2^2 + (E'x2)_1^2 + (E'x2)_2^2
    denominator = Ep1[:, 0]**2 + Ep1[:, 1]**2 + p2E[:, 0]**2 + p2E[:, 1]**2
    return numerator / denominator

def ransac_essential_matrix(pts1_pixel, pts2_pixel, K, n_iter=2000, threshold=0.005):
    K_inv = np.linalg.inv(K)
    pts1_norm = (K_inv @ np.column_stack((pts1_pixel, np.ones(len(pts1_pixel)))).T).T[:, :2]
    pts2_norm = (K_inv @ np.column_stack((pts2_pixel, np.ones(len(pts2_pixel)))).T).T[:, :2]

    best_inliers_count = 0
    best_E = None
    inlier_history = []
    
    for i in range(n_iter):
        idx = np.random.choice(len(pts1_norm), 8, replace=False)
        s1, s2 = pts1_norm[idx], pts2_norm[idx]
        
        # Estimate E from 8 samples
        E_cand, _ = cv2.findEssentialMat(s1, s2, np.eye(3), method=0)
        
        if E_cand is None or E_cand.shape != (3, 3):
            inlier_history.append(best_inliers_count)
            continue

        # Cheirality Check
        valid, R_cand, t_cand, _ = cv2.recoverPose(E_cand, s1, s2, np.eye(3))
        if valid == 0:
            inlier_history.append(best_inliers_count)
            continue
            
        errors = compute_sampson_distance(pts1_norm, pts2_norm, E_cand)
        inlier_mask = errors < threshold
        num_inliers = np.sum(inlier_mask)
        
        if num_inliers > best_inliers_count:
            best_inliers_count = num_inliers
            best_E = E_cand
        inlier_history.append(best_inliers_count)

    if best_inliers_count > 8:
        final_errors = compute_sampson_distance(pts1_norm, pts2_norm, best_E)
        final_mask = final_errors < threshold
        refined_E, _ = cv2.findEssentialMat(pts1_norm[final_mask], pts2_norm[final_mask], 
                                           np.eye(3), method=cv2.LMEDS)
        _, R, t, _ = cv2.recoverPose(refined_E, pts1_norm[final_mask], 
                                     pts2_norm[final_mask], np.eye(3))
        return R, t, refined_E, final_mask, inlier_history

    return None, None, None, None, inlier_history

pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
R, t, E_final, inlier_mask, history = ransac_essential_matrix(pts1, pts2, K)

def draw_epipolar_lines(img1, img2, pts1, pts2, E, K, num_lines=20):
    Ki = np.linalg.inv(K)
    F = Ki.T @ E @ Ki #This is the fundamental matrix
    
    # Pick random inliers for clear visualization 
    idx = np.random.choice(len(pts1), min(num_lines, len(pts1)), replace=False)
    p1, p2 = pts1[idx], pts2[idx]

    # Lines in img2 for points in img1
    lines2 = cv2.computeCorrespondEpilines(p1.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
    # Lines in img1 for points in img2
    lines1 = cv2.computeCorrespondEpilines(p2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

    def draw_on_img(img, lines, pts):
        r, c = img.shape[:2]
        out = img.copy()
        for l, p in zip(lines, pts):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -l[2]/l[1]])
            x1, y1 = map(int, [c, -(l[2]+l[0]*c)/l[1]])
            cv2.line(out, (x0, y0), (x1, y1), color, 2)
            cv2.circle(out, tuple(p.astype(int)), 8, color, -1)
        return out

    return draw_on_img(img1, lines1, p1), draw_on_img(img2, lines2, p2)


epip_img1, epip_img2 = draw_epipolar_lines(img1, img2, pts1[inlier_mask], pts2[inlier_mask], E_final, K)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(cv2.cvtColor(epip_img1, cv2.COLOR_BGR2RGB)); ax1.set_title("Epipolar Lines Img 1"); ax1.axis('off')
ax2.imshow(cv2.cvtColor(epip_img2, cv2.COLOR_BGR2RGB)); ax2.set_title("Epipolar Lines Img 2"); ax2.axis('off')
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history, color='tab:green', linewidth=2)
plt.title('RANSAC Convergence')
plt.xlabel('Iterations'); plt.ylabel('Inliers'); plt.grid(True, alpha=0.3)
plt.show()

inlier_matches = [good_matches[i] for i, val in enumerate(inlier_mask) if val]
img_inliers = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=2)
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_inliers, cv2.COLOR_BGR2RGB)); plt.title(f'RANSAC Inliers: {len(inlier_matches)} found'); plt.axis('off')
plt.show()