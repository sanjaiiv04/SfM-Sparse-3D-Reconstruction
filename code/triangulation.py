import viser
import time
import numpy as np
import cv2
import viser.transforms as tf
from ransac import R, t, inlier_mask, pts1, pts2
from intrinsics import K
from sift import img1, img2

def triangulate_points(pts1, pts2, R, t, K):
    # Projection matrices
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t.reshape(3, 1)))

    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Converting from Homogeneous to 3D by dividing with W
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]
    
    return points_3d.T

# using the inliers from the RANSAC algorithm
pts1_inliers = pts1[inlier_mask]
pts2_inliers = pts2[inlier_mask]

# Perform Triangulation
cloud_3d = triangulate_points(pts1_inliers, pts2_inliers, R, t, K)

# Calculating the median distance to remove points that "exploded" to infinity
dist = np.linalg.norm(cloud_3d, axis=1)
threshold = np.percentile(dist, 95) # Keep the closest 95% of points
filter_mask = dist < threshold

filtered_cloud = cloud_3d[filter_mask]
filtered_pts1 = pts1_inliers[filter_mask]

colors = []
for pt in filtered_pts1:
    x, y = int(pt[0]), int(pt[1])
    colors.append(img1[y, x])
colors = np.array(colors)[:, ::-1] / 255.0 # BGR to RGB and normalize to 0-1

server = viser.ViserServer()

# Adding the Point Cloud
server.scene.add_point_cloud(
    "/points",
    points=filtered_cloud,
    colors=colors,
    point_size=0.02
)

# Add Camera 1 
server.scene.add_camera_frustum(
    "/cameras/c1",
    fov=2 * np.arctan(K[0, 2] / K[0, 0]),
    aspect=img1.shape[1] / img1.shape[0],
    scale=0.15,
    image=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB),
)

# Add Camera 2 (at R, t)
# We need the World Pose 
T_world_cam2 = np.eye(4)
T_world_cam2[:3, :3] = R.T
T_world_cam2[:3, 3] = -R.T @ t.flatten()

server.scene.add_camera_frustum(
    "/cameras/c2",
    fov=2 * np.arctan(K[0, 2] / K[0, 0]),
    aspect=img2.shape[1] / img2.shape[0],
    scale=0.15,
    image=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB),
    wxyz=tf.SO3.from_matrix(T_world_cam2[:3, :3]).wxyz,
    position=T_world_cam2[:3, 3],
)



print("Viser is running! Visit the URL shown in the terminal.")
print(f"Visualizing {len(filtered_cloud)} inlier points.")

while True:
    time.sleep(1)