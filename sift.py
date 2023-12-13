import cv2
import numpy as np
from sklearn.cluster import DBSCAN

vid = "./unlabeled/6.hevc"
cap = cv2.VideoCapture(vid)

feature_params = dict(maxCorners=100000, qualityLevel=0.05, minDistance=1, blockSize=9)
lk_params = dict(winSize=(15, 15), maxLevel=10, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
height, width = old_frame.shape[:2]

# Calculate dimensions for the square
square_size = min(int(height/3), int(width/3))  # Size of the square
top_left_x = int(width/2 - square_size/2)
top_left_y = int(height/2 - square_size/2)

# Create a mask for the square
mask = np.zeros_like(old_frame)
mask[top_left_y:top_left_y + square_size, top_left_x:top_left_x + square_size] = 255
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)  # Apply mask here

scaling_factor = 30
percent = 1

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    good_new = p1[st == 1]
    good_old = p0[st == 1]


    # Calculate velocities
    velocities = np.sqrt((good_new[:, 0] - good_old[:, 0])**2 + (good_new[:, 1] - good_old[:, 1])**2)
    velocity_threshold = np.percentile(velocities, percent)

    fast_corners = []

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        if velocities[i] >= velocity_threshold:
            fast_corners.append((a, b))

    if fast_corners:
        # Use DBSCAN clustering to find the densest area of fast-moving corners
        clustering = DBSCAN(eps=50, min_samples=10).fit(fast_corners)
        labels = clustering.labels_

        # Find the largest cluster
        largest_cluster_index = max(set(labels), key=list(labels).count)
        largest_cluster = np.array(fast_corners)[labels == largest_cluster_index]

        if len(largest_cluster) > 0:
            (x, y), radius = cv2.minEnclosingCircle(largest_cluster)
            frame = cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)  # Draw enclosing circle
            frame = cv2.circle(frame, (int(x), int(y)), 10, (0, 0, 255), -1)  # Draw center dot

    cv2.imshow('frame', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
