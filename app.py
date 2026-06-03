import threading
import time
import os
import urllib.request
from datetime import datetime

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import distance
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 81, 13, 311, 291, 402, 14, 178]
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


def calculate_ear(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c)


def calculate_mar(mouth_pts):
    a = distance.euclidean(mouth_pts[1], mouth_pts[7])
    b = distance.euclidean(mouth_pts[2], mouth_pts[6])
    c = distance.euclidean(mouth_pts[3], mouth_pts[5])
    d = distance.euclidean(mouth_pts[0], mouth_pts[4])
    return (a + b + c) / (3.0 * d)


def compute_fatigue_score(ear, ear_thresh, yawn_count, drowsy_count, session_minutes):
    ear_score = max(0.0, (ear_thresh - ear) / ear_thresh) * 40
    event_rate = (yawn_count + drowsy_count * 2) / max(session_minutes, 0.5)
    event_score = min(event_rate * 5, 60)
    return min(int(ear_score + event_score), 100)


def draw_hud(frame, ear, mar, fatigue_score, is_drowsy, is_yawning):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (280, 118), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    def put(text, y, color=(230, 230, 230)):
        cv2.putText(
            frame,
            text,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )

    ear_color = (0, 80, 255) if ear < 0.25 else (0, 220, 100)
    mar_color = (0, 140, 255) if mar > 0.55 else (0, 220, 100)

    put(f"EAR: {ear:.3f}", 26, ear_color)
    put(f"MAR: {mar:.3f}", 52, mar_color)
    put(
        f"Fatigue: {fatigue_score}%",
        78,
        (0, 0, 255)
        if fatigue_score > 60
        else (0, 200, 255)
        if fatigue_score > 30
        else (0, 220, 100),
    )
    put(datetime.now().strftime("%H:%M:%S"), 104)

    if is_drowsy:
        cv2.rectangle(frame, (0, h - 72), (w, h), (0, 0, 180), -1)
        cv2.putText(
            frame,
            "DROWSINESS DETECTED",
            (max(12, w // 2 - 235), h - 26),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if is_yawning:
        y1 = h - 144 if is_drowsy else h - 72
        y2 = h - 72 if is_drowsy else h
        cv2.rectangle(frame, (0, y1), (w, y2), (140, 80, 0), -1)
        cv2.putText(
            frame,
            "YAWN DETECTED",
            (max(12, w // 2 - 150), y1 + 46),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 220, 100),
            2,
            cv2.LINE_AA,
        )

    return frame


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


class FaceLandmarkDetector:
    def __init__(self):
        self.uses_solutions_api = hasattr(mp, "solutions")
        if self.uses_solutions_api:
            self.detector = mp.solutions.face_mesh.FaceMesh(
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            ensure_model()
            options = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.mp_image = mp.Image
            self.mp_image_format = mp.ImageFormat
            self.detector = vision.FaceLandmarker.create_from_options(options)

    def detect(self, rgb_frame):
        if self.uses_solutions_api:
            results = self.detector.process(rgb_frame)
            return results.multi_face_landmarks or []

        image = self.mp_image(image_format=self.mp_image_format.SRGB, data=rgb_frame)
        results = self.detector.detect_for_video(image, int(time.time() * 1000))
        return results.face_landmarks or []


def normalize_landmarks(face_landmarks):
    return face_landmarks.landmark if hasattr(face_landmarks, "landmark") else face_landmarks


class FatigueVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.ear_thresh = 0.25
        self.mar_thresh = 0.55
        self.consec_frames = 20
        self.yawn_frames = 10

        self.face_detector = FaceLandmarkDetector()

        self.logs = []
        self.drowsy_start = None
        self.yawn_count = 0
        self.drowsy_count = 0
        self.session_start = time.time()
        self.frame_counter = 0
        self.yawn_counter = 0
        self.yawn_cooldown = 0
        self.last_ear = 0.0
        self.last_mar = 0.0
        self.fatigue_score = 0
        self.status = "Waiting for a face"

    def update_settings(self, ear_thresh, mar_thresh, consec_frames, yawn_frames):
        with self.lock:
            self.ear_thresh = ear_thresh
            self.mar_thresh = mar_thresh
            self.consec_frames = consec_frames
            self.yawn_frames = yawn_frames

    def reset(self):
        with self.lock:
            self.logs = []
            self.drowsy_start = None
            self.yawn_count = 0
            self.drowsy_count = 0
            self.session_start = time.time()
            self.frame_counter = 0
            self.yawn_counter = 0
            self.yawn_cooldown = 0
            self.last_ear = 0.0
            self.last_mar = 0.0
            self.fatigue_score = 0
            self.status = "Waiting for a face"

    def snapshot(self):
        with self.lock:
            return {
                "logs": list(self.logs),
                "drowsy_count": self.drowsy_count,
                "yawn_count": self.yawn_count,
                "session_start": self.session_start,
                "last_ear": self.last_ear,
                "last_mar": self.last_mar,
                "fatigue_score": self.fatigue_score,
                "status": self.status,
            }

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_results = self.face_detector.detect(rgb_frame)

        with self.lock:
            ear_thresh = self.ear_thresh
            mar_thresh = self.mar_thresh
            consec_frames = self.consec_frames
            yawn_frames = self.yawn_frames

        is_drowsy = False
        is_yawning = False

        if face_results:
            face_landmarks = face_results[0]
            h, w, _ = img.shape
            lm = normalize_landmarks(face_landmarks)

            left_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE]
            mouth_pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in MOUTH]

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
            mar = calculate_mar(mouth_pts)

            with self.lock:
                self.last_ear = ear
                self.last_mar = mar

                if ear < ear_thresh:
                    self.frame_counter += 1
                    if self.frame_counter >= consec_frames:
                        is_drowsy = True
                        self.status = "Drowsiness detected. Please take a break."
                        if self.drowsy_start is None:
                            self.drowsy_start = time.time()
                            self.drowsy_count += 1
                            self.logs.append(
                                {
                                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "Event": "Drowsy",
                                    "EAR": f"{ear:.3f}",
                                    "MAR": "-",
                                }
                            )
                else:
                    if self.drowsy_start:
                        duration = time.time() - self.drowsy_start
                        self.logs.append(
                            {
                                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "Event": f"Awake after {duration:.1f}s",
                                "EAR": f"{ear:.3f}",
                                "MAR": "-",
                            }
                        )
                        self.drowsy_start = None
                    self.frame_counter = 0

                if self.yawn_cooldown > 0:
                    self.yawn_cooldown -= 1

                if mar > mar_thresh:
                    self.yawn_counter += 1
                    if self.yawn_counter >= yawn_frames and self.yawn_cooldown == 0:
                        is_yawning = True
                        self.status = "Yawning detected. Consider taking a break."
                        self.yawn_count += 1
                        self.yawn_counter = 0
                        self.yawn_cooldown = 60
                        self.logs.append(
                            {
                                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "Event": "Yawn",
                                "EAR": f"{ear:.3f}",
                                "MAR": f"{mar:.3f}",
                            }
                        )
                else:
                    self.yawn_counter = max(0, self.yawn_counter - 1)

                session_minutes = (time.time() - self.session_start) / 60.0
                self.fatigue_score = compute_fatigue_score(
                    ear,
                    ear_thresh,
                    self.yawn_count,
                    self.drowsy_count,
                    session_minutes,
                )

                if not is_drowsy and not is_yawning:
                    self.status = (
                        f"Fatigue score: {self.fatigue_score}%. Monitor closely."
                        if self.fatigue_score > 50
                        else "Alert"
                    )

            for pt in left_eye + right_eye:
                cv2.circle(img, pt, 2, (0, 255, 120), -1)
            for pt in mouth_pts:
                cv2.circle(img, pt, 2, (0, 200, 255), -1)
        else:
            with self.lock:
                self.status = "Waiting for a face"

        img = draw_hud(
            img,
            self.last_ear,
            self.last_mar,
            self.fatigue_score,
            is_drowsy,
            is_yawning,
        )
        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="Driver Fatigue Monitor", layout="wide")

st.markdown(
    """
<style>
    .main .block-container { padding-top: 1.5rem; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Driver Fatigue Monitor")
st.caption(
    "Real-time drowsiness and yawn detection using browser camera access, "
    "Eye Aspect Ratio (EAR), and Mouth Aspect Ratio (MAR)."
)

with st.sidebar:
    st.header("Fine-tuning")
    ear_thresh = st.slider("EAR Threshold (eye closure)", 0.10, 0.40, 0.25, 0.01)
    mar_thresh = st.slider("MAR Threshold (yawn)", 0.40, 0.80, 0.55, 0.01)
    consec_frames = st.slider("Consecutive frames for drowsy alert", 10, 40, 20, 1)
    yawn_frames = st.slider("Consecutive frames for yawn alert", 8, 30, 10, 1)
    st.divider()
    st.write("Click Start below and allow camera access in your browser.")
    refresh_clicked = st.button("Refresh stats and log")
    reset_clicked = st.button("Reset session")

rtc_configuration = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
}

try:
    turn_config = st.secrets.get("turn")
except Exception:
    turn_config = None

if turn_config:
    rtc_configuration["iceServers"].append(
        {
            "urls": turn_config["urls"],
            "username": turn_config["username"],
            "credential": turn_config["credential"],
        }
    )

ctx = webrtc_streamer(
    key="fatigue-monitor",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=FatigueVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

processor = ctx.video_processor
if processor:
    processor.update_settings(ear_thresh, mar_thresh, consec_frames, yawn_frames)
    if refresh_clicked:
        st.rerun()
    if reset_clicked:
        processor.reset()
        st.rerun()

status_placeholder = st.empty()
col1, col2, col3, col4, col5 = st.columns(5)
ear_metric = col1.empty()
mar_metric = col2.empty()
fatigue_metric = col3.empty()
drowsy_metric = col4.empty()
yawn_metric = col5.empty()
st.subheader("Session Log")
log_placeholder = st.empty()
download_placeholder = st.empty()


def render_dashboard(data):
    status = data["status"]
    if "Drowsiness" in status:
        status_placeholder.error(status)
    elif "Yawning" in status or data["fatigue_score"] > 50:
        status_placeholder.warning(status)
    else:
        status_placeholder.success(status)

    ear_metric.metric("EAR", f"{data['last_ear']:.3f}")
    mar_metric.metric("MAR", f"{data['last_mar']:.3f}")
    fatigue_metric.metric("Fatigue", f"{data['fatigue_score']}%")
    drowsy_metric.metric("Drowsy events", data["drowsy_count"])
    yawn_metric.metric("Yawn events", data["yawn_count"])

    if data["logs"]:
        df = pd.DataFrame(data["logs"])
        log_placeholder.dataframe(df, use_container_width=True)
        download_placeholder.download_button(
            "Export log as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"fatigue_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        log_placeholder.info(
            "No drowsy or yawn events logged yet. Keep the camera running, then click "
            "'Refresh stats and log' after testing an event."
        )
        download_placeholder.empty()


if processor:
    render_dashboard(processor.snapshot())
else:
    status_placeholder.info("Press Start to begin camera-based detection.")
    ear_metric.metric("EAR", "-")
    mar_metric.metric("MAR", "-")
    fatigue_metric.metric("Fatigue", "-")
    drowsy_metric.metric("Drowsy events", 0)
    yawn_metric.metric("Yawn events", 0)
    log_placeholder.info("Session events will appear here.")
