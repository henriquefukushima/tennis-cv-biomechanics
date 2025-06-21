import cv2
import numpy as np
import pickle
from pathlib import Path
from utils.img_process_utils import hsv_cv2hsv

# ===== Configurações =====
filename = 'myserve_1.mp4'

# kyrgios.mp4
#hsv_low = hsv_cv2hsv(60, 19, 60)
#hsv_high = hsv_cv2hsv(80, 71, 100)

# myserve.mp4
hsv_low  = hsv_cv2hsv( 70, 40, 50)      # H ≈ 70 °, S > 40 %, V > 50 %
hsv_high = hsv_cv2hsv(115,100,100)      # H ≈ 115 °, full Sat / Val

# ===== Parâmetros fixos =====
INPUT_DIR = Path('input_videos')
SERVE_DIR = Path('serve')
LAND_DIR = Path('landmarks')
FRAMES_DIR = Path('frames')
SERVE_DIR.mkdir(exist_ok=True)
LAND_DIR.mkdir(exist_ok=True)
FRAMES_DIR.mkdir(exist_ok=True)
print(f"Processando vídeo: {SERVE_DIR / filename}")

# ===== Inicialização do OpenCV =====
cap = cv2.VideoCapture(SERVE_DIR / filename)
paused = False
prev_mask = None
list_balls = []

# ===== Configuração do vídeo para Salvar =====
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para salvar o vídeo
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_path = SERVE_DIR / f'{filename.split(".")[0]}_processed.mp4'
out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
# =============================

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
    conts = []

    for c in contours:
        (x,y), r = cv2.minEnclosingCircle(c)
        if r > 5:
            cv2.circle(frame, (int(x),int(y)), int(r), (0,255,0), 2)
            conts += [(int(x), int(y))]
    list_balls.append(conts)

    out.write(frame)
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

# salva landmarks
with open(LAND_DIR / f'ball_{filename.split(".")[0]}.pkl', 'wb') as f:
    pickle.dump(list(list_balls), f)

cap.release()
out.release()
print(f"\nVídeo processado e salvo em: {out_path}")
cv2.destroyAllWindows()