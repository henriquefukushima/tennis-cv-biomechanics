import cv2
import numpy as np
import pickle
from pathlib import Path
from utils.img_process_utils import hsv_cv2hsv

# ===== Configurações =====
filename = 'myserve_landmarks.mp4'
#hsv_low = hsv_cv2hsv(60, 19, 60)
#hsv_high = hsv_cv2hsv(80, 71, 100)

#hsv_low = hsv_cv2hsv(140, 25, 40)
#hsv_high = hsv_cv2hsv(80, 71, 90)
# values you PASS to hsv_cv2hsv(h, s, v)  →  degrees / percent / percent
hsv_low  = hsv_cv2hsv( 70, 40, 50)      # H ≈ 70 °, S > 40 %, V > 50 %
hsv_high = hsv_cv2hsv(115,100,100)      # H ≈ 115 °, full Sat / Val

#hsv_low  = hsv_cv2hsv(30, 55, 40)   # H 30°,  S >55 %, V >40 %
#hsv_high = hsv_cv2hsv(55,100,100)   # H 55°,  S,V full

# ===== Parâmetros fixos =====
INPUT_DIR = Path('input_videos')
SERVE_DIR = Path('serve')
LAND_DIR = Path('landmarks')
FRAMES_DIR = Path('frames')
SERVE_DIR.mkdir(exist_ok=True)
LAND_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
print(f"Processando vídeo: {INPUT_DIR / filename}")

# ===== Inicialização do OpenCV =====
cap = cv2.VideoCapture(INPUT_DIR / filename)
paused = False
prev_mask = None

while cap.isOpened():
    
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("\nFim do vídeo ou erro ao ler o frame.")
            break
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, 2)   # remove speckles
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)   # fill small holes

    overlay = frame.copy()
    overlay[mask == 255] = (0, 255, 0) 

    
    # --------------- Filtro de Movimento ------------------------
    if prev_mask is None:
        prev_mask = mask.copy()
        continue                          # need two frames to compare

    diff = cv2.absdiff(mask, prev_mask)

    # any non-zero pixel in diff means "changed"
    _, motion = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY)

    # optional clean-up
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel, 1)

    # update for next frame BEFORE early-exit!
    prev_mask = mask.copy()

    # combine colour AND motion
    moving_ball_mask = cv2.bitwise_and(mask, motion)

    # quick debug view
    #cv2.imshow("Motion", motion)
    #cv2.imshow("Colour∧Motion", moving_ball_mask)

    # if nothing moved, skip expensive work
    if cv2.countNonZero(moving_ball_mask) == 0:
        key = cv2.waitKey(1) & 0xFF            # still listen for keys
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('s'):
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.imwrite(str(FRAMES_DIR / f'frame_{idx}.png'), frame)
            cv2.imwrite(str(FRAMES_DIR / f'mask_{idx}.png'), mask)
            print(f"📸 frame_{idx}.png & mask_{idx}.png salvos!")
        continue   
    # --------------------------------------------------------------------
    
    # Limpar máscara com contornos da bolinha de tênis
    contours, _ = cv2.findContours(moving_ball_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        (x,y), r = cv2.minEnclosingCircle(c)
        if r > 5:
            cv2.circle(frame, (int(x),int(y)), int(r), (0,255,0), 2)

    cv2.imshow('Moving Ball Mask', moving_ball_mask)
    #cv2.imshow('Overlay', overlay)
    cv2.imshow('Frame', frame)

    # --- Keyboard handling ----------------------------
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Saindo...")
        break
    elif key == ord('s'):
        idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        cv2.imwrite(str(FRAMES_DIR / f'frame_{idx}.png'), frame)
        cv2.imwrite(str(FRAMES_DIR / f'mask_{idx}.png'),  mask)
        print(f"📸 frame_{idx}.png & mask_{idx}.png salvos!")

    elif key == ord('p'):
        # Pausa o vídeo
        paused = not paused

cap.release()
cv2.destroyAllWindows()