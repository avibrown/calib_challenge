import cv2
import numpy as np


# Parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Load the video
vid = "./labeled/3.hevc"
cap = cv2.VideoCapture(vid)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Take first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw the velocity vectors
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # Draw line for velocity vector
        frame = cv2.line(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)
        # Draw a circle at the new position
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 121, 1), -1)

    cv2.imshow('Frame', frame)

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
