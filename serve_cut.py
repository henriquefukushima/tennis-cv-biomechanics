import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import pickle
from collections import deque
from pathlib import Path
from utils.logging_utils import show_progress

#===== Configurações =====
filename = 'myserve_1.mp4'
scale_percent = 80  
frameskip = 1   # Pular frames para reduzir a carga de processamento
rotate = True  # Rotacionar o vídeo 90 graus no sentido horário
#=========================

# Inicialização do MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose(static_image_mode=False,
                              model_complexity=2,
                              enable_segmentation=False,
                              smooth_landmarks=True,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)

# Criar uma pasta para salvar os vídeos dos saques se não existir
input_dir = Path('input_videos')
serve_dir = Path('serve')
land_dir = Path('landmarks')
os.makedirs(serve_dir, exist_ok=True)
os.makedirs(land_dir, exist_ok=True)

cap = cv2.VideoCapture(input_dir / filename)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec para salvar o vídeo
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Deque para armazenar os frames e landmarks
frames = deque(maxlen=fps * 4) 
landmarks_frames = deque(maxlen=fps * 4)

# Mínima visibilidade
min_visibility = 0.5

# Inicilizar flags para identificar o saque
condition1_met = condition2_met = condition3_met = False
serve_count = 0
out = None
post_condition3_frames = 0 # Contador de frames após a condição 3 ser verificada
current_frame = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("\nFim do vídeo ou erro ao ler o frame.")
        break

    current_frame += 1
    # Pular frames conforme o frameskip
    # Isso reduz a carga de processamento e acelera a detecção
    if current_frame % frameskip != 0:
        continue

    # Reescalando o frame para melhorar o processamento
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # Monitorando o progresso
    show_progress(current_frame, total_frames)

    if rotate:
        # Rotacionar o frame 90 graus no sentido horário
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
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
        if wrist_visible and landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y < landmarks[mp_pose.PoseLandmark.NOSE].y:
            condition1_met = True

        ## Condição 2: Cotovelo direito acima do ombro direito (laçada do saque)
        if condition1_met and elbow_visible and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y < landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y:
            condition2_met = True

        ## Condição 3: Cotovelo direito abaixo do ombro direito (finalização do saque)
        if condition2_met and elbow_visible and landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y > landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y:
            condition3_met = True

        if condition3_met:
            post_condition3_frames += 1 

        if condition3_met and out is None:
            serve_count += 1
            print(f'\nSaque detectado! Contagem: {serve_count}')
            out = cv2.VideoWriter(
                serve_dir / f'{filename.split(".")[0]}_{serve_count}.mp4',
                               fourcc=fourcc,
                               fps=fps,
                               frameSize=(frame.shape[1], frame.shape[0]))

        if out is not None:
            if post_condition3_frames >= fps * 2: # espera 2 segundos após a condição 3
                
                # salva landmarks
                with open(land_dir / f'{filename.split(".")[0]}_{serve_count}.pkl', 'wb') as f:
                    pickle.dump(list(landmarks_frames), f)

                while not len(frames) == 0:
                    out.write(frames.popleft())
                
                out.release()
                print(f'\nVídeo {serve_count} salvo!')

                # Reseta as condições
                out = None
                condition1_met = condition2_met = condition3_met = False
                post_condition3_frames = 0

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()