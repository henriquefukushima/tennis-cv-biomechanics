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
                              model_complexity=2,
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

# Deque para armazenar os frames e landmarks
frames = deque(maxlen=fps * 4) 
landmarks_frames = deque(maxlen=fps * 4)

# Mínima visibilidade
min_visibility = 0.5

# Inicilizar flags para identificar o saque
condition1_met = condition2_met = condition3_met = False
serve_count = 0

current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    current_frame += 1

    # Reescalando o frame para melhorar o processamento
    scale_percent = 30  
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Armazenar nas deques
        frames.append(frame)
        landmarks_frames.append(results.pose_landmarks)

        # Verificar condições para identificar o saque

        ## Checa visibilidade
        wrist_visible = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].visibility > min_visibility
        elbow_visible = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility > min_visibility
        shoulder_visible = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility > min_visibility

        ## Condição 1: Braço esquerdo acima do nariz
        if wrist_visible and landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[mp_pose.PoseLandmarks.NOSE].y:
            condition1_met = True

        ## Condição 2: Cotovelo direito acima do ombro direito (laçada do saque)
        if condition1_met and elbow_visible and landmarks[mp.pose.PoseLandmark.RIGHT_ELBOW].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y:
            condition2_met = True

        ## Condição 3: Cotovelo direito abaixo do ombro direito (finalização do saque)
        if condition2_met and elbow_visible and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y:
            condition3_met = True

        if condition3_met:
            serve_count += 1
            print(f'Saque detectado! Contagem: {serve_count}')
            condition1_met = condition2_met = condition3_met = False


    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()