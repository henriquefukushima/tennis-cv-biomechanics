import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import pickle
from collections import deque
from pathlib import Path

# Inicialização do MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose(static_image_mode=False,
                              model_complexity=1,
                              enable_segmentation=False,
                              smooth_landmarks=True,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)

# Criar uma pasta para salvar os vídeos dos saques se não existir
serve_dir = Path('serve')
land_dir = Path('landmarks')
os.makedirs(serve_dir, exist_ok=True)
os.makedirs(land_dir, exist_ok=True)

filename = 'myserve_1.mp4'
cap = cv2.VideoCapture(serve_dir / filename)
fps = int(cap.get(cv2.CAP_PROP_FPS))

while cap.isOpened():
    ret, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)

    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()