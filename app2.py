import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
from ultralytics import YOLO
import cvzone
import math
import numpy as np

# Load the YOLO model
model = YOLO("ppe.pt")

classNames = ['Gloves', 'Helmet', 'Non-Helmet', 'Person', 'Shoes', 'Vest', 'bare-arms']

# Set the page title and configure the layout
# st.set_page_config(page_title="PPE Detection with YOLOv8", layout="wide")

# Load custom HTML template with background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://wallpapers.com/images/featured-full/safety-pictures-yxhdianfjuydk3zt.jpg");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stSidebar"] > div:.st-emotion-cache-1cypcdb eczjsme11 {{
background-image: url("https://wallpapers.com/images/featured-full/safety-pictures-yxhdianfjuydk3zt.jpg");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Create a sidebar for input elements
with st.sidebar:
    st.title("Configuration")

    # Select input type: Webcam or File Upload
    input_type = st.radio("Select Input Type:", ["Webcam", "File Upload"])

    # Adjust confidence threshold
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

    # Create a "Detect Objects" button
    detect_button = st.button("Detect Objects")

    # Create a "Stop Webcam" button (initially hidden)
    stop_webcam_button = st.empty()
    webcam_running = False  # Flag to track if webcam is running

    if input_type == "File Upload":
        st.write("Using File Upload")

        # File Upload
        uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "mp4"])

# Main content area
st.title("PPE Detection Model")


# Define a VideoTransformer class for processing the video stream
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.cap = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=confidence_threshold)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                w, h = x2 - x1, y2 - y1
                x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = (math.ceil(box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
        return img

# Use webrtc_streamer to create the WebRTC video streamer
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Rest of your code for handling file uploads
# ...
