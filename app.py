import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
import time


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Rim & Wheel AI Detector",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🚗 Rim & Wheel AI Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'YOLO Pose Detection • 4 Keypoints'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = YOLO("best.pt")

    return model


try:

    model = load_model()

except Exception as e:

    st.error(
        "Could not load best.pt. "
        "Make sure best.pt is in the same folder as app.py."
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Settings")

input_type = st.sidebar.radio(
    "Select Input",
    [
        "Image",
        "Video"
    ]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.0,
    value=0.25,
    step=0.05
)

image_size = st.sidebar.selectbox(
    "Image Size",
    [
        320,
        480,
        640,
        800,
        1024
    ],
    index=2
)


# =========================================================
# IMAGE MODE
# =========================================================

if input_type == "Image":

    st.header("📷 Image Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        # -------------------------------------------------
        # Original
        # -------------------------------------------------

        with col1:

            st.subheader("Original Image")

            st.image(
                image,
                use_container_width=True
            )

        # -------------------------------------------------
        # Detect
        # -------------------------------------------------

        if st.button(
            "🔍 Detect",
            use_container_width=True
        ):

            start_time = time.time()

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=image_size,
                verbose=False
            )

            processing_time = (
                time.time() - start_time
            )

            result = results[0]

            annotated = result.plot()

            # -------------------------------------------------
            # Result
            # -------------------------------------------------

            with col2:

                st.subheader(
                    "Detection Result"
                )

                st.image(
                    annotated,
                    channels="BGR",
                    use_container_width=True
                )

            # -------------------------------------------------
            # FPS
            # -------------------------------------------------

            fps = (
                1 / processing_time
                if processing_time > 0
                else 0
            )

            col_a, col_b, col_c = st.columns(3)

            with col_a:

                st.metric(
                    "Processing Time",
                    f"{processing_time:.3f} sec"
                )

            with col_b:

                st.metric(
                    "Estimated FPS",
                    f"{fps:.2f}"
                )

            # -------------------------------------------------
            # Detection information
            # -------------------------------------------------

            st.subheader(
                "📊 Detection Information"
            )

            if (
                result.boxes is not None
                and len(result.boxes) > 0
            ):

                class_ids = (
                    result.boxes.cls
                    .cpu()
                    .numpy()
                )

                confidences = (
                    result.boxes.conf
                    .cpu()
                    .numpy()
                )

                for i, (
                    class_id,
                    conf
                ) in enumerate(
                    zip(
                        class_ids,
                        confidences
                    )
                ):

                    class_name = (
                        model.names[
                            int(class_id)
                        ]
                    )

                    st.write(
                        f"**Object {i + 1}:** "
                        f"{class_name}  "
                        f"— Confidence: "
                        f"{conf:.2%}"
                    )

                # -------------------------------------------------
                # Keypoints
                # -------------------------------------------------

                if result.keypoints is not None:

                    st.subheader(
                        "📍 Keypoints"
                    )

                    keypoints = (
                        result.keypoints.xy
                        .cpu()
                        .numpy()
                    )

                    for object_index, points in enumerate(
                        keypoints
                    ):

                        st.write(
                            f"**Object "
                            f"{object_index + 1}**"
                        )

                        for point_index, point in enumerate(
                            points
                        ):

                            x = point[0]
                            y = point[1]

                            st.write(
                                f"Keypoint "
                                f"{point_index + 1}: "
                                f"X={x:.1f}, "
                                f"Y={y:.1f}"
                            )

            else:

                st.warning(
                    "No wheel or rim detected."
                )


# =========================================================
# VIDEO MODE
# =========================================================

else:

    st.header("🎥 Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ]
    )

    if uploaded_video is not None:

        if st.button(
            "🚀 Start Detection",
            use_container_width=True
        ):

            # -------------------------------------------------
            # Save uploaded video
            # -------------------------------------------------

            temp_input = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            )

            temp_input.write(
                uploaded_video.read()
            )

            temp_input.close()

            # -------------------------------------------------
            # Open video
            # -------------------------------------------------

            cap = cv2.VideoCapture(
                temp_input.name
            )

            if not cap.isOpened():

                st.error(
                    "Could not open video."
                )

                st.stop()

            # -------------------------------------------------
            # Video information
            # -------------------------------------------------

            original_fps = (
                cap.get(
                    cv2.CAP_PROP_FPS
                )
            )

            total_frames = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            width = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            height = int(
                cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            st.info(
                f"Original Video FPS: "
                f"{original_fps:.2f}"
            )

            # -------------------------------------------------
            # Output video
            # -------------------------------------------------

            output_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4"
            ).name

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            out = cv2.VideoWriter(
                output_path,
                fourcc,
                original_fps,
                (width, height)
            )

            # -------------------------------------------------
            # UI
            # -------------------------------------------------

            frame_placeholder = st.empty()

            progress_bar = st.progress(0)

            fps_placeholder = st.empty()

            detection_placeholder = st.empty()

            # -------------------------------------------------
            # FPS calculation
            # -------------------------------------------------

            frame_count = 0

            start_time = time.time()

            # -------------------------------------------------
            # Process frames
            # -------------------------------------------------

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                # ---------------------------------------------
                # YOLO
                # ---------------------------------------------

                results = model.predict(
                    source=frame,
                    conf=confidence,
                    imgsz=image_size,
                    verbose=False
                )

                result = results[0]

                # ---------------------------------------------
                # Draw boxes + keypoints
                # ---------------------------------------------

                annotated_frame = (
                    result.plot()
                )

                # ---------------------------------------------
                # Frame count
                # ---------------------------------------------

                frame_count += 1

                elapsed = (
                    time.time()
                    - start_time
                )

                current_fps = (
                    frame_count / elapsed
                    if elapsed > 0
                    else 0
                )

                # ---------------------------------------------
                # Detection count
                # ---------------------------------------------

                detection_count = 0

                if (
                    result.boxes is not None
                ):

                    detection_count = len(
                        result.boxes
                    )

                # ---------------------------------------------
                # Add FPS to frame
                # ---------------------------------------------

                cv2.putText(
                    annotated_frame,
                    f"FPS: {current_fps:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # ---------------------------------------------
                # Add detection count
                # ---------------------------------------------

                cv2.putText(
                    annotated_frame,
                    f"Objects: {detection_count}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                # ---------------------------------------------
                # Write video
                # ---------------------------------------------

                out.write(
                    annotated_frame
                )

                # ---------------------------------------------
                # Display
                # ---------------------------------------------

                frame_placeholder.image(
                    annotated_frame,
                    channels="BGR",
                    use_container_width=True
                )

                # ---------------------------------------------
                # Progress
                # ---------------------------------------------

                if total_frames > 0:

                    progress = (
                        frame_count
                        / total_frames
                    )

                    progress_bar.progress(
                        min(
                            progress,
                            1.0
                        )
                    )

                # ---------------------------------------------
                # FPS UI
                # ---------------------------------------------

                fps_placeholder.metric(
                    "🚀 Processing FPS",
                    f"{current_fps:.2f}"
                )

                detection_placeholder.metric(
                    "🔍 Detected Objects",
                    detection_count
                )

            # -------------------------------------------------
            # Release
            # -------------------------------------------------

            cap.release()

            out.release()

            # -------------------------------------------------
            # Final statistics
            # -------------------------------------------------

            total_time = (
                time.time()
                - start_time
            )

            average_fps = (
                frame_count / total_time
                if total_time > 0
                else 0
            )

            st.success(
                "✅ Video processing completed!"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Total Frames",
                    frame_count
                )

            with col2:

                st.metric(
                    "Average FPS",
                    f"{average_fps:.2f}"
                )

            with col3:

                st.metric(
                    "Original FPS",
                    f"{original_fps:.2f}"
                )

            # -------------------------------------------------
            # Download
            # -------------------------------------------------

            with open(
                output_path,
                "rb"
            ) as video_file:

                st.download_button(
                    label="⬇️ Download Result Video",
                    data=video_file,
                    file_name=(
                        "rim_detection_result.mp4"
                    ),
                    mime="video/mp4",
                    use_container_width=True
                )

            # -------------------------------------------------
            # Cleanup
            # -------------------------------------------------

            try:

                os.unlink(
                    temp_input.name
                )

            except:

                pass