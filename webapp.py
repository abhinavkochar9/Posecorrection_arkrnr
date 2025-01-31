import streamlit as st
import cv2
import tempfile
import os
import glob
import mediapipe as mp
import numpy as np
import time
from scipy.signal import argrelextrema
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Title of the Web App
st.title("Exercise Tutorial & Feedback App")

# Path to Reference Videos Directory
video_dir = "/Users/abhinavkochar/Desktop/Pose correction webapp/Videos"

# Fetch all .mov files from the directory
reference_videos = sorted(glob.glob(os.path.join(video_dir, "*.mov")))
video_options = {os.path.basename(video).replace(".mov", ""): video for video in reference_videos}

# Dropdown to select exercise
selected_exercise = st.sidebar.selectbox("Select an Exercise", list(video_options.keys()))

# Upload Function in Sidebar
st.sidebar.write("### Upload Your Exercise Video")
uploaded_video = st.sidebar.file_uploader("Upload your video file", type=["mp4", "avi", "mov"])

st.write("### Reference Video and Uploaded Video Side by Side")
col1, col2 = st.columns(2)

if uploaded_video is not None:
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_video.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_video.read())
    
    if st.button("Start Feedback"):
        st.write("### Feedback Analysis in Progress...")
        
        # Initialize MediaPipe Pose
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mp_drawing = mp.solutions.drawing_utils
        
        # Define key points indices for shoulders, elbows, and wrists
        keypoint_indices = [11, 12, 13, 14, 15, 16]  # Left shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist
        keypoint_names = ["Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist"]
        
        def extract_selected_keypoints(frame):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                keypoints = np.array([[results.pose_landmarks.landmark[i].x,
                                       results.pose_landmarks.landmark[i].y,
                                       results.pose_landmarks.landmark[i].z] for i in keypoint_indices]).flatten()
                return results.pose_landmarks, keypoints
            return None, None
        
        cap_ref = cv2.VideoCapture(video_options[selected_exercise])
        cap_user = cv2.VideoCapture(temp_path)
        
        ref_placeholder = col1.empty()
        user_placeholder = col2.empty()
        feedback_placeholder = st.empty()
        
        similarity_scores = {joint: [] for joint in keypoint_names}
        time_series = []
        frame_count = 0
        
        while cap_ref.isOpened() and cap_user.isOpened():
            ret_ref, frame_ref = cap_ref.read()
            ret_user, frame_user = cap_user.read()
            
            if not ret_ref or not ret_user:
                break
            
            ref_landmarks, ref_keypoints = extract_selected_keypoints(frame_ref)
            user_landmarks, user_keypoints = extract_selected_keypoints(frame_user)
            
            if ref_keypoints is not None and user_keypoints is not None:
                real_time_similarity = {}
                for i, joint in enumerate(keypoint_names):
                    joint_ref = ref_keypoints[i * 3:(i + 1) * 3]
                    joint_user = user_keypoints[i * 3:(i + 1) * 3]
                    joint_similarity = 1 - cosine(joint_ref, joint_user)
                    similarity_scores[joint].append(joint_similarity * 100)
                    real_time_similarity[joint] = joint_similarity * 100
                
                feedback_placeholder.write("### Real-Time Similarity Scores:")
                feedback_text = ""
                for joint, score in real_time_similarity.items():
                    feedback_text += f"{joint}: {score:.2f}/100\n"
                feedback_placeholder.text(feedback_text)
                
                time_series.append(time.time())
            
            if user_landmarks:
                for i in keypoint_indices:
                    h, w, _ = frame_user.shape
                    cx, cy = int(user_landmarks.landmark[i].x * w), int(user_landmarks.landmark[i].y * h)
                    cv2.circle(frame_user, (cx, cy), 5, (0, 255, 0), -1)
            
            ref_placeholder.image(frame_ref, channels="BGR")
            user_placeholder.image(frame_user, channels="BGR")
            
            frame_count += 1
            time.sleep(0.05)
        
        cap_ref.release()
        cap_user.release()
        
        st.write("### Feedback Analysis Complete")
        
        # Generate pose similarity graphs for each keypoint
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, joint in enumerate(keypoint_names):
            axes[i].plot(time_series, similarity_scores[joint], label=f"{joint} Similarity", color="blue")
            axes[i].set_xlabel("Time (s)")
            axes[i].set_ylabel("Similarity Score")
            axes[i].set_title(f"Pose Similarity Over Time - {joint}")
            axes[i].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
