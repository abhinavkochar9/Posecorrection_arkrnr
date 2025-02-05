import streamlit as st
import cv2
import os
import glob
import mediapipe as mp
import numpy as np
import time
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from fastdtw import fastdtw
import plotly.graph_objects as go

# -----------------------------------------------------------
#                       CONSTANTS
# -----------------------------------------------------------

VIDEO_DIR = "/Users/abhinavkochar/Desktop/Pose correction webapp/video/video/processed_video/skeletal_video"
PATIENT_ROOT = "/Users/abhinavkochar/Desktop/Pose correction webapp/CleanedData_final"
MIN_REPS_FOR_ALIGNMENT = 3

KEYPOINT = {
    "indices": [11, 12, 13, 14, 15, 16],
    "names": [
        "Left Shoulder",
        "Right Shoulder",
        "Left Elbow",
        "Right Elbow",
        "Left Wrist",
        "Right Wrist"
    ]
}

FEEDBACK_THRESHOLDS = {
    "Shoulder": 85,
    "Elbow": 80,
    "Wrist": 75
}

FEEDBACK_PROMPTS = {
    "Shoulder": {
        "low": "Keep your shoulders stable and avoid excessive movement",
        "medium": "Try to maintain shoulder alignment throughout the exercise",
        "high": "Great shoulder control! Maintain this form"
    },
    "Elbow": {
        "low": "Focus on maintaining proper elbow angles during movement",
        "medium": "Your elbow movement is improving, aim for more consistency",
        "high": "Excellent elbow control, keep it up!"
    },
    "Wrist": {
        "low": "Pay attention to wrist positioning, avoid excessive bending",
        "medium": "Your wrist movement is good, aim for more stability",
        "high": "Perfect wrist control, maintain this precision"
    }
}

# -----------------------------------------------------------
#                 HELPER FUNCTIONS
# -----------------------------------------------------------

mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose_estimator():
    return mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

pose_estimator = load_pose_estimator()

def normalize_coordinates(landmarks, width, height):
    """Divide all coordinates by shoulder width for scale normalization."""
    points = np.array([[lm.x * width, lm.y * height] for lm in landmarks])
    shoulder_width = np.linalg.norm(points[0] - points[1])  # left & right shoulders
    return (points / shoulder_width).flatten()

def calculate_similarity(a, b):
    """Similarity = (1 / (1 + Euclidean dist)) * 100."""
    dist = euclidean(a, b)
    return (1 / (1 + dist)) * 100

def resize_with_padding(frame, target_size):
    """Resize a frame with black padding to match (width, height)."""
    h, w = frame.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))

    top = (target_size[1] - new_h) // 2
    bottom = (target_size[1] - new_h + 1) // 2
    left = (target_size[0] - new_w) // 2
    right = (target_size[0] - new_w + 1) // 2
    return cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

def process_video(video_path):
    """Read frames from a .mp4 file, run MediaPipe Pose, store frames/keypoints/timestamps."""
    cap = cv2.VideoCapture(video_path)
    frames, keypoints, timestamps = [], [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_estimator.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            chosen_lm = [results.pose_landmarks.landmark[i] for i in KEYPOINT["indices"]]
            kps = normalize_coordinates(chosen_lm, frame.shape[1], frame.shape[0])
            keypoints.append(kps)
            frames.append(frame)
            timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

    cap.release()
    return {
        "frames": frames,
        "keypoints": np.array(keypoints),
        "timestamps": np.array(timestamps)
    }

def align_sequences(ref_data, user_data):
    """Use FastDTW to align the two sets of keypoint sequences."""
    _, path = fastdtw(ref_data["keypoints"], user_data["keypoints"], dist=euclidean)
    return path

def detect_exercise_phases(keypoints):
    """Identify peaks in velocity to find 'reps' or transitions."""
    velocities = np.linalg.norm(np.diff(keypoints, axis=0), axis=1)
    mean_vel = np.mean(velocities)
    peaks, _ = find_peaks(velocities, distance=10, prominence=mean_vel)
    return peaks

def generate_feedback_prompts(similarity_scores):
    """Create feedback for each joint based on average similarity across frames."""
    feedback = {}
    for joint, scores in similarity_scores.items():
        avg_score = np.mean(scores)
        jt = joint.split()[1]  # e.g. "Shoulder"
        threshold = FEEDBACK_THRESHOLDS[jt]
        low_threshold = threshold * 0.7

        if avg_score < low_threshold:
            feedback[joint] = {
                "score": avg_score,
                "message": FEEDBACK_PROMPTS[jt]["low"],
                "status": "Needs Improvement"
            }
        elif avg_score < threshold:
            feedback[joint] = {
                "score": avg_score,
                "message": FEEDBACK_PROMPTS[jt]["medium"],
                "status": "Good"
            }
        else:
            feedback[joint] = {
                "score": avg_score,
                "message": FEEDBACK_PROMPTS[jt]["high"],
                "status": "Excellent"
            }
    return feedback

def find_deflections(time_axis, similarity_values, threshold=70):
    """
    Return intervals (start_time, end_time) where average similarity < threshold.
    """
    deflection_intervals = []
    in_deflection = False
    start_time = None

    for i in range(len(similarity_values)):
        if similarity_values[i] < threshold and not in_deflection:
            in_deflection = True
            start_time = time_axis[i]
        elif similarity_values[i] >= threshold and in_deflection:
            in_deflection = False
            end_time = time_axis[i]
            deflection_intervals.append((start_time, end_time))

    if in_deflection:
        deflection_intervals.append((start_time, time_axis[-1]))

    return deflection_intervals

def replay_deflection_interval(ref_data, user_data, alignment_map,
                               start_time, end_time, deflection_idx=1, 
                               extra_buffer=2.0):
    """
    Replay the portion of the video (Reference vs. User) from
    (start_time - extra_buffer) to (end_time + extra_buffer).
    """
    st.markdown(f"### Replaying Deflection Interval {deflection_idx}")
    timeline = ref_data["timestamps"]
    actual_start = max(0, start_time - extra_buffer)
    actual_end = min(timeline[-1], end_time + extra_buffer)

    st.write(f"Showing from {actual_start:.2f}s to {actual_end:.2f}s "
             f"(±{extra_buffer}s buffer).")

    col_left, col_right = st.columns(2)
    ref_placeholder = col_left.empty()
    user_placeholder = col_right.empty()

    # Gather indices in [actual_start, actual_end]
    idxs = np.where((timeline >= actual_start) & (timeline <= actual_end))[0]
    if len(idxs) == 0:
        st.warning("No frames found in this interval.")
        return

    for ref_idx in idxs:
        user_idx = alignment_map.get(ref_idx, ref_idx)
        user_idx = min(user_idx, len(user_data["frames"]) - 1)

        ref_frame = ref_data["frames"][ref_idx]
        user_frame = user_data["frames"][user_idx]
        user_frame = resize_with_padding(user_frame, (ref_frame.shape[1], ref_frame.shape[0]))

        ref_placeholder.image(ref_frame, channels="BGR", 
                             caption=f"Ref: {timeline[ref_idx]:.2f}s")
        user_placeholder.image(user_frame, channels="BGR", 
                              caption=f"User: ~{timeline[ref_idx]:.2f}s")
        time.sleep(0.03)  # playback speed

# -----------------------------------------------------------
#             STREAMLIT APPLICATION CODE
# -----------------------------------------------------------

st.title("Exercise Feedback: Two Graphs, Progress Bar, and Single Timeline")

# -- Session State --
if "analysis_data" not in st.session_state:
    st.session_state["analysis_data"] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "selected_interval" not in st.session_state:
    st.session_state["selected_interval"] = None

# -- Sidebar Inputs --
st.sidebar.header("Exercise Selection")
ref_videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
exercise = st.sidebar.selectbox(
    "Choose Exercise",
    [os.path.basename(v).replace(".mp4", "") for v in ref_videos]
)

st.sidebar.header("Patient Selection")
patients = [
    d for d in os.listdir(PATIENT_ROOT)
    if os.path.isdir(os.path.join(PATIENT_ROOT, d))
]
patient = st.sidebar.selectbox("Choose Patient", patients)

deflection_threshold = st.sidebar.slider(
    "Deflection Threshold (%)",
    min_value=0, max_value=100, value=50, step=1,
    help="Any average similarity below this threshold is considered a deflection."
)

analyze_button = st.sidebar.button("Analyze Exercise")

def data_matches_current(data_dict):
    if not data_dict:
        return False
    return (
        data_dict["exercise"] == exercise
        and data_dict["patient"] == patient
        and data_dict["threshold"] == deflection_threshold
    )

# If user changes selection or hits "Analyze," reset
if analyze_button or not data_matches_current(st.session_state["analysis_data"]):
    st.session_state["analysis_data"] = None
    st.session_state["analysis_done"] = False
    st.session_state["selected_interval"] = None

# 1) Process & Align if needed
if not st.session_state["analysis_done"]:
    if analyze_button:
        # Step A: Locate videos
        ref_candidates = [v for v in ref_videos if exercise in v]
        if not ref_candidates:
            st.error("No reference video found for the selected exercise.")
            st.stop()
        ref_path = ref_candidates[0]

        patient_dir = os.path.join(PATIENT_ROOT, patient, exercise)
        patient_files = [f for f in os.listdir(patient_dir) if f.endswith(".mp4")]
        if not patient_files:
            st.error("No patient video found for the selected patient/exercise.")
            st.stop()

        patient_video_path = os.path.join(patient_dir, patient_files[0])

        # Step B: Process videos
        with st.spinner("Processing reference video..."):
            ref_data = process_video(ref_path)
        with st.spinner("Processing patient video..."):
            user_data = process_video(patient_video_path)

        # Step C: Align sequences and check reps
        with st.spinner("Aligning movements..."):
            alignment_path = align_sequences(ref_data, user_data)
            ref_peaks = detect_exercise_phases(ref_data["keypoints"])
            user_peaks = detect_exercise_phases(user_data["keypoints"])
            if len(ref_peaks) < MIN_REPS_FOR_ALIGNMENT or len(user_peaks) < MIN_REPS_FOR_ALIGNMENT:
                st.error("Not enough reps detected for reliable alignment.")
                st.stop()
        alignment_map = {r: u for r, u in alignment_path}

        # Step D: Pre-compute per-frame joint similarity
        similarity = {joint: [] for joint in KEYPOINT["names"]}
        num_frames = len(ref_data["frames"])
        user_total_frames = len(user_data["frames"])

        for ref_idx in range(num_frames):
            user_idx = alignment_map.get(ref_idx, ref_idx)
            user_idx = min(user_idx, user_total_frames - 1)

            for j, joint in enumerate(KEYPOINT["names"]):
                ref_kp = ref_data["keypoints"][ref_idx][j*2 : (j+1)*2]
                user_kp = user_data["keypoints"][user_idx][j*2 : (j+1)*2]
                score = calculate_similarity(ref_kp, user_kp)
                similarity[joint].append(score)

        # Step E: Find average similarity & deflection intervals
        all_joints = list(similarity.keys())
        similarity_array = np.array([similarity[j] for j in all_joints])  # shape (6, num_frames)
        avg_similarity_per_frame = np.mean(similarity_array, axis=0)
        ref_times = ref_data["timestamps"][:num_frames]
        deflection_intervals = []
        deflection_frames = set()

        if len(ref_times) > 0:
            deflection_intervals = find_deflections(ref_times, avg_similarity_per_frame, deflection_threshold)
            # Convert those intervals to a set of frame indices
            for (start_t, end_t) in deflection_intervals:
                idxs = np.where((ref_times >= start_t) & (ref_times <= end_t))[0]
                deflection_frames.update(idxs)

        # Step F: Full side-by-side playback with progress bar
        st.write("### Full Side-by-Side Playback with Progress")
        prog_bar = st.progress(0, text="Starting playback...")
        col_left, col_right = st.columns(2)
        ref_placeholder = col_left.empty()
        user_placeholder = col_right.empty()
        feedback_placeholder = st.empty()

        for ref_idx in range(num_frames):
            user_idx = alignment_map.get(ref_idx, ref_idx)
            user_idx = min(user_idx, user_total_frames - 1)

            ref_frame = ref_data["frames"][ref_idx]
            user_frame = user_data["frames"][user_idx]
            user_frame = resize_with_padding(user_frame, (ref_frame.shape[1], ref_frame.shape[0]))

            # Frame-level feedback
            frame_feedback = []
            for j, joint in enumerate(KEYPOINT["names"]):
                score = similarity[joint][ref_idx]
                jt = joint.split()[1]
                thr = FEEDBACK_THRESHOLDS[jt]
                low_thr = thr * 0.7
                if score < low_thr:
                    frame_feedback.append(f"⚠️ {joint}: Needs Improvement ({score:.1f}%)")
                elif score < thr:
                    frame_feedback.append(f"🟡 {joint}: Good ({score:.1f}%)")
                else:
                    frame_feedback.append(f"✅ {joint}: Excellent ({score:.1f}%)")

            # Display frames
            ref_placeholder.image(ref_frame, channels="BGR")
            user_placeholder.image(user_frame, channels="BGR")

            feedback_placeholder.markdown(
                "**Frame Feedback:**\n" + "\n".join(frame_feedback)
            )

            progress_val = int((ref_idx+1)/num_frames * 100)
            if ref_idx in deflection_frames:
                prog_bar.progress(progress_val, text=f"Deflection! Frame {ref_idx+1}/{num_frames}")
            else:
                prog_bar.progress(progress_val, text=f"Playing... Frame {ref_idx+1}/{num_frames}")

            time.sleep(0.03)  # playback speed

        # Step G: Store in session_state
        st.session_state["analysis_data"] = {
            "exercise": exercise,
            "patient": patient,
            "threshold": deflection_threshold,
            "ref_data": ref_data,
            "user_data": user_data,
            "alignment_map": alignment_map,
            "similarity": similarity,
            "avg_similarity_per_frame": avg_similarity_per_frame,
            "deflection_intervals": deflection_intervals
        }
        st.session_state["analysis_done"] = True
        st.success("Playback done! Scroll down for graphs and the single timeline.")
    else:
        st.info("Click 'Analyze Exercise' to begin.")
        st.stop()

else:
    # Already analyzed
    pass

# 2) Display results if we have them
if st.session_state["analysis_done"]:
    data = st.session_state["analysis_data"]
    ref_data = data["ref_data"]
    alignment_map = data["alignment_map"]
    similarity = data["similarity"]
    avg_similarity = data["avg_similarity_per_frame"]
    deflection_intervals = data["deflection_intervals"]

    # Summaries
    st.subheader("Joint Feedback Summary")
    final_feedback = generate_feedback_prompts(similarity)
    for joint, fb in final_feedback.items():
        score = fb["score"]
        with st.expander(f"{joint} - {fb['status']} ({score:.1f}%)"):
            st.write(fb["message"])
            st.progress(min(score/100, 1.0))

    # Graph 1: All Joint Similarities
    st.subheader("Graph 1: Individual Joint Similarities")
    fig1 = go.Figure()
    ref_times = ref_data["timestamps"][:len(avg_similarity)]
    n_frames = len(ref_times)

    # Plot each joint
    for joint in KEYPOINT["names"]:
        sims = similarity[joint][:n_frames]
        fig1.add_trace(go.Scatter(
            x=ref_times,
            y=sims,
            mode="lines",
            name=joint,
            hovertemplate=f"{joint}<br>Time: %{{x:.2f}}s<br>Score: %{{y:.1f}}%"
        ))

    # Mark reference "rep" boundaries
    ref_peaks = detect_exercise_phases(ref_data["keypoints"])
    for i, p in enumerate(ref_peaks):
        if p < n_frames:
            fig1.add_vline(
                x=ref_times[p],
                line_dash="dot",
                annotation_text=f"Rep {i+1}",
                annotation_position="top right"
            )

    # Add red rectangles for deflection intervals (based on average)
    for (start_t, end_t) in deflection_intervals:
        fig1.add_vrect(
            x0=start_t, x1=end_t,
            fillcolor="red", opacity=0.1,
            layer="below", line_width=0
        )

    fig1.update_layout(
        title="Per-Joint Similarities Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Similarity (%)",
        hovermode="x unified",
        height=500
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Graph 2: Average Similarity
    st.subheader("Graph 2: Average Similarity Over Time")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=ref_times,
        y=avg_similarity[:n_frames],
        mode="lines",
        name="Average Similarity",
        line=dict(width=3, color="black"),
        hovertemplate="Time: %{x:.2f}s<br>Score: %{y:.1f}%"
    ))

    # Mark reps
    for i, p in enumerate(ref_peaks):
        if p < n_frames:
            fig2.add_vline(
                x=ref_times[p],
                line_dash="dot",
                annotation_text=f"Rep {i+1}",
                annotation_position="top right"
            )

    # Deflections in red
    for (start_t, end_t) in deflection_intervals:
        fig2.add_vrect(
            x0=start_t, x1=end_t,
            fillcolor="red", opacity=0.2,
            layer="below", line_width=0
        )

    fig2.update_layout(
        title=f"Average Similarity vs. Time (Deflection < {data['threshold']}%)",
        xaxis_title="Time (s)",
        yaxis_title="Similarity (%)",
        hovermode="x unified",
        height=450
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Single horizontal timeline for deflections
    st.subheader("Single Horizontal Timeline of Deflections")
    if len(ref_times) > 0:
        total_duration = ref_times[-1]
        fig_timeline = go.Figure(layout=dict(
            xaxis=dict(range=[0, total_duration], showgrid=False),
            yaxis=dict(range=[0, 1], showgrid=False),
            height=150,
            margin=dict(l=40, r=40, t=40, b=40),
            title="Deflection Intervals on a Single Timeline"
        ))

        # Base rectangle (normal region)
        fig_timeline.add_shape(
            type="rect",
            x0=0, x1=total_duration,
            y0=0, y1=1,
            fillcolor="lightgreen",
            line_width=0,
            opacity=0.3
        )

        # Red rectangles for each deflection
        for i, (start_t, end_t) in enumerate(deflection_intervals, start=1):
            fig_timeline.add_shape(
                type="rect",
                x0=start_t, x1=end_t,
                y0=0, y1=1,
                fillcolor="red",
                opacity=0.5,
                line_width=0
            )
            # Optional label in the middle
            mid_t = (start_t + end_t) / 2
            fig_timeline.add_annotation(
                x=mid_t, y=0.5,
                text=f"Deflection {i}",
                showarrow=False,
                font=dict(color="white", size=12)
            )

        fig_timeline.update_xaxes(title_text="Time (s)")
        fig_timeline.update_yaxes(visible=False)
        st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.warning("No reference frames to display timeline.")

    # Deflection intervals & replay
    st.subheader("Deflection Intervals & Replays")
    if deflection_intervals:
        for i, (start_t, end_t) in enumerate(deflection_intervals, start=1):
            st.write(f"**Interval {i}**: {start_t:.2f}s -> {end_t:.2f}s")
            if st.button(f"Replay Interval {i}"):
                st.session_state["selected_interval"] = (i, start_t, end_t)
    else:
        st.write("No deflection intervals detected.")

# 3) Replay interval if selected
if st.session_state["selected_interval"] is not None:
    i, st_time, en_time = st.session_state["selected_interval"]
    st.session_state["selected_interval"] = None
    data = st.session_state["analysis_data"]
    replay_deflection_interval(
        ref_data=data["ref_data"],
        user_data=data["user_data"],
        alignment_map=data["alignment_map"],
        start_time=st_time,
        end_time=en_time,
        deflection_idx=i,
        extra_buffer=2.0
    )
    st.stop()