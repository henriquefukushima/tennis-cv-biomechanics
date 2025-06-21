import numpy as np
import pdb
import cv2 
from scipy.spatial import distance

def hsv_cv2hsv(h, s, v):
    """
    Convert HSV given in *degrees* (H 0-360) and *percent* (S, V 0-100)
    to OpenCV's 8-bit HSV (H 0-179, S,V 0-255).

    Returns
    -------
    np.ndarray  shape (3,)  dtype uint8
    """
    return np.array([
        int(h / 2),               # Hue: 0-360 → 0-179  (OpenCV uses half-degrees)
        int(s / 100 * 255),       # Sat: 0-100% → 0-255
        int(v / 100 * 255)        # Val: 0-100% → 0-255
    ], dtype=np.uint8)

def draw_shadows(frame, landmarks, idx, bp, len_):
    idx0 = max(idx - len_, 0)
    h, w, _ = frame.shape
    line_points = [(int(i.landmark[bp].x * w), int(i.landmark[bp].y * h)) for i in landmarks[idx0: idx]]

    for i in range(1, len(line_points)):
        thickness = int(np.sqrt(20 / float(i + 1)) * 2.5)
        cv2.line(frame, line_points[i - 1], line_points[i], (255, 255, 30), thickness=thickness, lineType=cv2.LINE_AA)

    return frame

def draw_ball_shadow(frame, ball, idx):
    idx0 = max(idx - 20, 0)
    linepoints = [i for i in ball[idx0:idx]]

    for i in range(1, len(linepoints)):
        thickness = int(np.sqrt(30 / float(i + 1)) * 2.5)

        bs0 = linepoints[i - 1]
        bs1 = linepoints[i]
        if len(bs0) > 0 and len(bs1) > 0:
            for p0 in bs0:
                min_dist = 10
                p1_min = None
                for p1 in bs1:
                    dist = distance.euclidean(p0, p1)
                    if dist < min_dist:
                        p1_min = p1
                if p1_min is not None:
                    cv2.line(frame, p0, p1_min, (255, 255, 30), thickness=thickness)

    return frame