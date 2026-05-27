import cv2
import streamlit as st
import mediapipe as mp
from scipy.spatial import distance
import pygame
import threading
import time
import pandas as pd
import os
import urllib.request
import numpy as np
from datetime import datetime

def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

MOUTH = [61, 81, 13, 311, 291, 402, 14, 178]

def calculate_mar(mouth_pts):
    # Vertical distances
    A = distance.euclidean(mouth_pts[1], mouth_pts[7])   # 81 <-> 178
    B = distance.euclidean(mouth_pts[2], mouth_pts[6])   # 13 <-> 14
    C = distance.euclidean(mouth_pts[3], mouth_pts[5])   # 311 <-> 402
    # Horizontal distance
    D = distance.euclidean(mouth_pts[0], mouth_pts[4])   # 61 <-> 291
    return (A + B + C) / (3.0 * D)

def compute_fatigue_score(ear, ear_thresh, yawn_count, drowsy_count, session_minutes):
    ear_score  = max(0.0, (ear_thresh - ear) / ear_thresh) * 40   # 0-40 pts
    event_rate = (yawn_count + drowsy_count * 2) / max(session_minutes, 0.5)
    event_score = min(event_rate * 5, 60)                         # 0-60 pts
    return min(int(ear_score + event_score), 100)

try:
    pygame.mixer.init()
except Exception as e:
    print(f"Warning initializing pygame mixer: {e}")

_alert_lock = threading.Lock()
_alert_active = False

def play_alert(sound_file="alert.mp3"):
    global _alert_active
    with _alert_lock:
        if _alert_active:
            return
        _alert_active = True
    try:
        if os.path.exists(sound_file):
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play(loops=-1)
        else:
            sample_rate = 22050
            duration    = 0.5
            freq        = 880
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            wave = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
            wave = np.stack([wave, wave], axis=-1)
            sound = pygame.sndarray.make_sound(wave)
            sound.play(loops=-1)
    except Exception as e:
        print(f"Audio error: {e}")

def stop_alert():
    global _alert_active
    try:
        pygame.mixer.music.stop()
        pygame.mixer.stop()
    except Exception:
        pass
    with _alert_lock:
        _alert_active = False

LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading face landmarker model…"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def normalize_landmarks(landmarks):
    return landmarks.landmark if hasattr(landmarks, "landmark") else landmarks

if hasattr(mp, "solutions"):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh    = mp_face_mesh.FaceMesh(refine_landmarks=True)

    def get_face_landmarks(rgb_frame, timestamp_ms):
        results = face_mesh.process(rgb_frame)
        return results.multi_face_landmarks
else:
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    ensure_model()
    options   = vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    face_mesh = vision.FaceLandmarker.create_from_options(options)

    def get_face_landmarks(rgb_frame, timestamp_ms):
        image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = face_mesh.detect_for_video(image, timestamp_ms)
        return results.face_landmarks

st.set_page_config(page_title="Drowsiness & Yawn Detection", layout="wide")

st.markdown("""
<style>
    header[data-testid="stHeader"] {
        height: 2.5rem !important;
        min-height: 2.5rem !important;
    }
    .main .block-container {
        padding-top: 0 !important;
        margin-top: -4rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Driver Fatigue Monitor")
st.caption("Real-time drowsiness + yawn detection using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)")

with st.sidebar:
    st.header("Fine-tuning")
    ear_thresh    = st.slider("EAR Threshold (eye closure)", 0.10, 0.40, 0.25, 0.01)
    mar_thresh    = st.slider("MAR Threshold (yawn)",        0.40, 0.80, 0.55, 0.01)
    consec_frames = st.slider("Consecutive frames for drowsy alert", 10, 40, 20, 1)
    yawn_frames   = st.slider("Consecutive frames for yawn alert",    8, 30, 10, 1)
    st.divider()
    st.info("Lower EAR → more sensitive to eye closure")
    st.info("Higher MAR → mouth must open wider to trigger yawn")
    st.write("Audio alert triggers on events.")
    st.write("All events are logged below.")

defaults = {
    "logs":           [],
    "drowsy_start":   None,
    "alert_played":   False,
    "yawn_count":     0,
    "drowsy_count":   0,
    "session_start":  None,
    "frame_counter":  0,
    "yawn_counter":   0,
    "yawn_cooldown":  0,          
    "last_ear":       0.0,
    "last_mar":       0.0,
    "fatigue_score":  0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

btn_col, _ = st.columns([1, 3])
with btn_col:
    run_toggle = st.toggle("▶ Start Detection", value=False)

status_placeholder = st.empty()

col_video, col_stats = st.columns([2, 1])

with col_video:
    frame_placeholder = st.empty()

with col_stats:
    st.subheader("Live Stats")
    ear_metric      = st.metric("EAR",        "–")
    mar_metric      = st.metric("MAR",        "–")
    fatigue_metric  = st.metric("Fatigue %",  "–")
    st.divider()
    drowsy_metric   = st.metric("Drowsy events",  st.session_state.drowsy_count)
    yawn_metric     = st.metric("Yawn events",    st.session_state.yawn_count)

    def update_sidebar_metrics(ear, mar, score):
        ear_metric.metric("EAR",       f"{ear:.3f}",  delta=None)
        mar_metric.metric("MAR",       f"{mar:.3f}",  delta=None)
        fatigue_metric.metric("Fatigue %", f"{score}%", delta=None)
        drowsy_metric.metric("Drowsy events", st.session_state.drowsy_count)
        yawn_metric.metric("Yawn events",    st.session_state.yawn_count)

def draw_hud(frame, ear, mar, fatigue_score, is_drowsy, is_yawning):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (260, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def put(text, y, color=(220, 220, 220)):
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    ear_color = (0, 80, 255) if ear < 0.25 else (0, 220, 100)
    mar_color = (0, 140, 255) if mar > 0.55 else (0, 220, 100)

    put(f"EAR : {ear:.3f}", 24,  ear_color)
    put(f"MAR : {mar:.3f}", 48,  mar_color)
    put(f"Fatigue: {fatigue_score}%", 72,
        (0, 0, 255) if fatigue_score > 60 else (0, 200, 255) if fatigue_score > 30 else (0, 220, 100))
    put(f"{datetime.now().strftime('%H:%M:%S')}", 96)

    if is_drowsy:
        cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, "DROWSINESS DETECTED", (w // 2 - 230, h - 25),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    if is_yawning:
        cv2.rectangle(frame, (0, h - 140 if is_drowsy else h - 70), (w, h - 70 if is_drowsy else h), (140, 80, 0), -1)
        cv2.putText(frame, "YAWN DETECTED", (w // 2 - 145, (h - 105) if is_drowsy else (h - 25)),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 220, 100), 2, cv2.LINE_AA)

    return frame

if run_toggle:
    if st.session_state.session_start is None:
        st.session_state.session_start = time.time()

    cap = cv2.VideoCapture(0)

    while run_toggle and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read from webcam.")
            break

        rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.time() * 1000)
        results      = get_face_landmarks(rgb_frame, timestamp_ms)

        is_drowsy  = False
        is_yawning = False

        if results:
            for face_landmarks in results:
                h, w, _ = frame.shape
                lm = normalize_landmarks(face_landmarks)

                left_eye  = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE]
                right_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE]
                ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0

                mouth_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in MOUTH]
                mar = calculate_mar(mouth_pts)

                st.session_state.last_ear = ear
                st.session_state.last_mar = mar

                if ear < ear_thresh:
                    st.session_state.frame_counter += 1
                    if st.session_state.frame_counter >= consec_frames:
                        is_drowsy = True
                        if not st.session_state.alert_played:
                            if st.session_state.drowsy_start is None:
                                st.session_state.drowsy_start = time.time()
                                ts = time.strftime('%Y-%m-%d %H:%M:%S')
                                st.session_state.logs.append({"Timestamp": ts, "Event": "Drowsy", "EAR": f"{ear:.3f}", "MAR": "–"})
                                st.session_state.drowsy_count += 1
                            threading.Thread(target=play_alert, daemon=True).start()
                            st.session_state.alert_played = True
                else:
                    if st.session_state.drowsy_start:
                        dur = time.time() - st.session_state.drowsy_start
                        ts  = time.strftime('%Y-%m-%d %H:%M:%S')
                        st.session_state.logs.append({"Timestamp": ts, "Event": f"Awake (after {dur:.1f}s)", "EAR": f"{ear:.3f}", "MAR": "–"})
                        st.session_state.drowsy_start = None
                        st.session_state.alert_played = False
                        stop_alert()
                    st.session_state.frame_counter = 0

                if st.session_state.yawn_cooldown > 0:
                    st.session_state.yawn_cooldown -= 1

                if mar > mar_thresh:
                    st.session_state.yawn_counter += 1
                    if (st.session_state.yawn_counter >= yawn_frames
                            and st.session_state.yawn_cooldown == 0):
                        is_yawning = True
                        ts = time.strftime('%Y-%m-%d %H:%M:%S')
                        st.session_state.logs.append({"Timestamp": ts, "Event": "Yawn", "EAR": f"{ear:.3f}", "MAR": f"{mar:.3f}"})
                        st.session_state.yawn_count  += 1
                        st.session_state.yawn_counter = 0
                        st.session_state.yawn_cooldown = 60   # ~2 s cooldown at 30 fps

                        if not st.session_state.alert_played:
                            threading.Thread(target=play_alert, daemon=True).start()
                            # Auto-stop yawn alert after 1.5 s so it doesn't loop
                            def _stop_after():
                                time.sleep(1.5)
                                stop_alert()
                            threading.Thread(target=_stop_after, daemon=True).start()
                else:
                    st.session_state.yawn_counter = max(0, st.session_state.yawn_counter - 1)

                session_minutes = (time.time() - st.session_state.session_start) / 60.0
                st.session_state.fatigue_score = compute_fatigue_score(
                    ear, ear_thresh,
                    st.session_state.yawn_count,
                    st.session_state.drowsy_count,
                    session_minutes,
                )

                for pt in left_eye + right_eye:
                    cv2.circle(frame, pt, 2, (0, 255, 120), -1)
                for pt in mouth_pts:
                    cv2.circle(frame, pt, 2, (0, 200, 255), -1)

        frame = draw_hud(
            frame,
            st.session_state.last_ear,
            st.session_state.last_mar,
            st.session_state.fatigue_score,
            is_drowsy,
            is_yawning,
        )

        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        update_sidebar_metrics(
            st.session_state.last_ear,
            st.session_state.last_mar,
            st.session_state.fatigue_score,
        )

        if is_drowsy:
            status_placeholder.error("🔴 DROWSY — Please take a break!")
        elif is_yawning:
            status_placeholder.warning("🟡 Yawning detected — Consider taking a break.")
        elif st.session_state.fatigue_score > 50:
            status_placeholder.warning(f"🟠 Fatigue score: {st.session_state.fatigue_score}% — Monitor closely.")
        else:
            status_placeholder.success("🟢 Alert")
    cap.release()

else:
    stop_alert()
    frame_placeholder.info("Toggle **Start Detection** above to begin.")

if st.session_state.logs:
    st.divider()
    st.subheader("📋 Session Log")
    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇Export log as CSV",
        data=csv,
        file_name=f"fatigue_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    drowsy_events = len(df[df["Event"] == "Drowsy"])
    yawn_events   = len(df[df["Event"] == "Yawn"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Drowsy Events", drowsy_events)
    col2.metric("Total Yawn Events",   yawn_events)
    if st.session_state.session_start:
        elapsed = (time.time() - st.session_state.session_start) / 60
        col3.metric("Session Duration", f"{elapsed:.1f} min")
