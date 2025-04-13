import cv2
import streamlit as st
import mediapipe as mp
from scipy.spatial import distance
import pygame
import threading
import time
import pandas as pd

# EAR calculation
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

pygame.mixer.init()

def play_alert():
    try:
        pygame.mixer.music.load("alert.mp3")  # Use your MP3 file
        pygame.mixer.music.play(loops=-1)  # Loop the sound indefinitely
        print("Alert sound played successfully")
    except Exception as e:
        print(f"Error playing sound: {e}")

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Streamlit GUI
st.set_page_config(page_title="Drowsiness Detection", layout="centered")
st.title("Drowsiness Detection System")

# Sidebar Controls
run_toggle = st.sidebar.toggle("Start Detection", value=False)
ear_thresh = st.sidebar.slider("EAR Threshold", 0.1, 0.4, 0.25, 0.01)
st.sidebar.write("🔊 Audio alert will trigger when drowsiness is detected.")
st.sidebar.write("📋 Logging enabled.")

# Initialize logging
if "logs" not in st.session_state:
    st.session_state.logs = []
if "drowsy_start" not in st.session_state:
    st.session_state.drowsy_start = None
if "alert_played" not in st.session_state:
    st.session_state.alert_played = False  # Flag to track if alert has already played

# Drowsiness Constants
CONSEC_FRAMES = 20
frame_counter = 0

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Stream video
frame_placeholder = st.empty()

# Video capture only if toggle is on
if run_toggle:
    cap = cv2.VideoCapture(0)
    while run_toggle and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
                right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < ear_thresh:
                    frame_counter += 1
                    if frame_counter >= CONSEC_FRAMES:
                        # Drowsiness detected, show alert
                        cv2.putText(frame, "DROWSINESS DETECTED!", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                        # Check if alert has been played
                        if not st.session_state.alert_played:
                            # Log drowsiness start time
                            if st.session_state.drowsy_start is None:
                                st.session_state.drowsy_start = time.time()
                                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                                st.session_state.logs.append({"Timestamp": timestamp, "Status": "Drowsy"})

                            # Play alert sound in a new thread
                            threading.Thread(target=play_alert, daemon=True).start()

                            # Mark alert as played
                            st.session_state.alert_played = True

                else:
                    frame_counter = 0
                    # If the person is no longer drowsy, stop the alert sound and reset the flag
                    if st.session_state.drowsy_start:
                        duration = time.time() - st.session_state.drowsy_start
                        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                        st.session_state.logs.append({"Timestamp": timestamp, "Status": f"Awake (after {duration:.1f}s)"})
                        st.session_state.drowsy_start = None
                        st.session_state.alert_played = False  # Reset the alert flag when the person wakes up
                        pygame.mixer.music.stop()  # Stop the alert sound when awake

                # Draw eye landmarks
                for point in left_eye + right_eye:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()

# Show session log
if st.session_state.logs:
    st.subheader("📋 Session Log")
    df = pd.DataFrame(st.session_state.logs)
    st.dataframe(df, use_container_width=True)