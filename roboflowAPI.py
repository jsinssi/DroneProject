import cv2
import subprocess
import numpy as np
import av
import asyncio
from aiortc import VideoStreamTrack
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import StreamConfig, VideoMetadata

print("VERSION: SDK COMPATIBILITY FIX")

# --- CUSTOM VIDEO TRACK ---
class PiCameraTrack(VideoStreamTrack):
    def __init__(self, resolution=(1280, 720), fps=30):
        super().__init__()
        self.width, self.height = resolution
        self.fps = fps
        self.frame_size = int(self.width * self.height * 1.5)
        self.running = True
        
        print(f"[Camera] Starting rpicam-vid at {self.width}x{self.height}...")
        
        cmd = [
            "rpicam-vid", "--nopreview", "-t", "0", "--inline",
            "--width", str(self.width), "--height", str(self.height),
            "--framerate", str(self.fps), "--codec", "yuv420", "-o", "-"
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

    async def recv(self):
        if not self.running:
            raise av.AudioFrame() # Signal end of stream

        loop = asyncio.get_event_loop()
        # Run the blocking read in a separate thread
        raw_image = await loop.run_in_executor(None, self.proc.stdout.read, self.frame_size)
        
        if len(raw_image) != self.frame_size:
            print("Warning: Camera stream incomplete")
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return av.VideoFrame.from_ndarray(blank, format="bgr24")

        # Convert YUV -> BGR
        yuv = np.frombuffer(raw_image, dtype=np.uint8).reshape((int(self.height * 1.5), self.width))
        frame_bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        
        # Create WebRTC Frame
        video_frame = av.VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        pts, time_base = await self.next_timestamp()
        video_frame.pts = pts
        video_frame.time_base = time_base
        
        return video_frame

    def stop(self):
        self.running = False
        if self.proc:
            self.proc.terminate()

# --- CUSTOM SOURCE ADAPTER ---
class PiCameraSource:
    def __init__(self, resolution=(1280, 720), fps=30):
        self.track = PiCameraTrack(resolution, fps)
        
    async def configure_peer_connection(self, pc):
        # Attach the video track to the WebRTC connection
        pc.addTrack(self.track)

    def get_initialization_params(self, stream_config):
        # The SDK asks for metadata about the source
        return {
            "width": self.track.width,
            "height": self.track.height,
            "fps": self.track.fps
        }

    async def cleanup(self):
        # Must be async for the SDK
        self.track.stop()

# --- MAIN CODE ---

print("[Main] Initializing Roboflow Client...")
client = InferenceHTTPClient.init(
    api_url="https://serverless.roboflow.com",
    api_key="RUOJDNiiVKtE5T8Kro2x"
)

print("[Main] Opening Camera Source...")
source = PiCameraSource(resolution=(1280, 720), fps=30)

print("[Main] Configuring Stream...")
config = StreamConfig(
    stream_output=["output_image"],
    data_output=["output_deduplicated", "count_objects", "predictions"],
    requested_plan="webrtc-gpu-medium",
    requested_region="us",
)

print("[Main] Connecting to Server (This may take 30 seconds)...")
session = client.webrtc.stream(
    source=source,
    workflow="detect-count-and-visualize-4",
    workspace="parking-project-kwqj7",
    image_input="image",
    config=config
)

@session.on_frame
def show_frame(frame, metadata):
    cv2.imshow("Workflow Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    print(f"Data received for Frame {metadata.frame_id}")

try:
    print("[Main] Starting Session...")
    session.run()
except KeyboardInterrupt:
    print("\n[Main] Stopping...")
finally:
    # Ensure camera process is killed
    source.track.stop()
    cv2.destroyAllWindows()
