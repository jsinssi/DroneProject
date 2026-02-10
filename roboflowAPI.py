# libcamerify python cameraroboflow.py
import cv2
import sys
import os
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

# --- SECURITY NOTE ---
# Please rotate your API Key (RUOJ...) in the Roboflow dashboard settings.
# Anyone reading this chat history can currently use your quota.
API_KEY = "RUOJDNiiVKtE5T8Kro2x" 

# Initialize client
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key=API_KEY
)

# Configure video source
# On Pi Camera 3, we often need libcamerify (which you used correctly!)
# If 1280x720 is too slow, try (640, 480)
source = WebcamSource(resolution=(1280, 720))

# Configure streaming options
# REMOVED: processing_timeout (caused the error)
config = StreamConfig(
    stream_output=["output_image"],
    data_output=["output_deduplicated", "count_objects", "predictions"],
    requested_plan="webrtc-gpu-medium",
    requested_region="us",
)

# Create streaming session
session = client.webrtc.stream(
    source=source,
    workflow="detect-count-and-visualize-4",
    workspace="parking-project-kwqj7",
    image_input="image",
    config=config
)

# Handle incoming video frames
@session.on_frame
def show_frame(frame, metadata):
    # Only show popup window if a monitor is attached (avoids crash on SSH)
    if os.environ.get('DISPLAY') is not None:
        cv2.imshow("Workflow Output", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            session.close()

# Handle prediction data
@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    # Print counts to terminal
    if "count_objects" in data:
        print(f"Frame {metadata.frame_id}: Counts: {data['count_objects']}")
    else:
        print(f"Frame {metadata.frame_id}: {data}")

# Run the session
print("Stream starting... Press Ctrl+C to stop.")
try:
    session.run()
except KeyboardInterrupt:
    print("\nStopping...")
    session.close()
