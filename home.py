import streamlit as st
from pathlib import Path
import pickle
import cv2
import os

# ===== Configurações do Streamlit =====
st.set_page_config(page_title="Análise de Saques no Vôlei", 
                   layout="wide")

# ====== Constantes e Variáveis Globais ======
LANDMARKS_DIR = Path('landmarks')
SERVE_DIR = Path('serve')

if 'idx' not in st.session_state:
    st.session_state.idx = 0
if 'idx2' not in st.session_state:
    st.session_state.idx2 = 0
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False

def draw_sidebar(landmarks, landmarks2):

    expander_frame = st.sidebar.expander("Frame Controller", expanded=True)
    slider_play = expander_frame.empty()
    slider_play2 = expander_frame.empty()
    col1, col2 = expander_frame.columns(2)
    col3, col4 = st.sidebar.columns(2)
    if col1.button("-1", key="btn1"):
        st.session_state.idx = st.session_state.idx - 1 if st.session_state.idx > 0 else 0
    if col1.button("+1", key="btn2"):
        st.session_state.idx = st.session_state.idx + 1 if st.session_state.idx < len(landmarks) else st.session_state.idx
    if col2.button("-1", key="btn3"):
        st.session_state.idx2 = st.session_state.idx2 - 1 if st.session_state.idx2 > 0 else 0
    if col2.button("+1", key="btn4"):
        st.session_state.idx2 = st.session_state.idx2 + 1 if st.session_state.idx2 < len(landmarks2) else st.session_state.idx2

    if col1.button("Voltar", key="btn5"):
        st.session_state.is_playing = False
        st.session_state.idx = st.session_state.idx - 1 if st.session_state.idx > 0 else 0
        st.session_state.idx2 = st.session_state.idx2 - 1 if st.session_state.idx2 > 0 else 0

    if col2.button("Avançar", key="btn6"):
        st.session_state.is_playing = False
        st.session_state.idx = st.session_state.idx + 1 if st.session_state.idx < len(landmarks) else len(landmarks) - 1
        st.session_state.idx2 = st.session_state.idx2 + 1 if st.session_state.idx2 < len(landmarks2) else len(landmarks2) - 1

    if col3.button("Play", key="btn7"):
        st.session_state.is_playing = True
    if col4.button("Pause", key="btn8"):
        st.session_state.is_playing = False
    
    st.session_state.idx = slider_play.slider("Video 1", 0, len(landmarks) - 1, st.session_state.idx)
    st.session_state.idx2 = slider_play2.slider("Video 2", 0, len(landmarks2) - 1, st.session_state.idx2)

@st.cache_data()
def load_data(video, video2):
    land_file = LANDMARKS_DIR / f'landmarks_{video.split('.')[0]}.pkl'
    with open(land_file, 'rb') as f:
        landmarks = pickle.load(f)

    land_file2 = LANDMARKS_DIR / f'landmarks_{video2.split(".")[0]}.pkl'
    with open(land_file2, 'rb') as f:
        landmarks2 = pickle.load(f)
    
    ball_file = LANDMARKS_DIR / f'ball_{video.split(".")[0]}.pkl'
    with open(ball_file, 'rb') as f:
        ball = pickle.load(f)

    ball_file2 = LANDMARKS_DIR / f'ball_{video2.split(".")[0]}.pkl'
    with open(ball_file2, 'rb') as f:
        ball2 = pickle.load(f)

    return landmarks, landmarks2, ball, ball2

@st.cache_resource()
def load_video(video, video2):

    cap = {
        1: cv2.VideoCapture(str(SERVE_DIR / f'{video}.mp4')),
        2: cv2.VideoCapture(str(SERVE_DIR / f'{video2}.mp4'))
    }

    return cap

video_files = [i.split(".")[0].replace("ball_", "") for i in os.listdir("landmarks") if "ball" in i]
video_files.sort()

video = st.sidebar.selectbox("Selecione o primeiro vídeo", video_files)
video2 = st.sidebar.selectbox("Selecione o segundo vídeo", video_files, index=len(video_files) - 1)

render_ball = st.checkbox("Exibir bola?")

body_parts = {
    "RIGHT UPPER": 16,
    "LEFT UPPER": 15,
    "RIGHT LOWER": 28,
    "LEFT LOWER": 27,
}
bp = st.multiselect("Marcações do corpo", body_parts.keys(), default=["RIGHT UPPER"])

cap = load_video(video, video2)
lands_data, landas_data2, ball_data, ball_data2 = load_data(video, video2)

draw_sidebar(lands_data, landas_data2)

col1, col2, col3 = st.columns(3)
ph1 = col1.empty()
cont1 = ph1.container()
ph2 = col2.empty()
cont2 = ph2.container()
ph3 = col3.empty()
cont3 = ph3.container()

if not st.session_state["is_playing"]:
    cap[1].set(cv2.CAP_PROP_POS_FRAMES, st.session_state.idx)
    cap[2].set(cv2.CAP_PROP_POS_FRAMES, st.session_state.idx2)
    
    ret1, frame1 = cap[1].read()
    ret2, frame2 = cap[2].read()
    h1, w1, _ = frame1.shape
    h2, w2, _ = frame2.shape
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    cont1.image(frame1, use_container_width=True)
    cont2.image(frame2, use_container_width=True)

while st.session_state["is_playing"]:
    st.session_state.idx += 1 if st.session_state.idx < len(lands_data) - 1 else 0
    st.session_state.idx2 += 1 if st.session_state.idx2 < len(landas_data2) - 1 else 0

    cap[1].set(cv2.CAP_PROP_POS_FRAMES, st.session_state.idx)
    cap[2].set(cv2.CAP_PROP_POS_FRAMES, st.session_state.idx2)
    ret1, frame1 = cap[1].read()
    ret2, frame2 = cap[2].read()
    h1, w1, _ = frame1.shape
    h2, w2, _ = frame2.shape
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    with ph1.container() as p:
        st.image(frame1, use_container_width=True)
    with ph2.container() as p:
        st.image(frame2, use_container_width=True)