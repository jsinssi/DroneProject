# import the necessary packages
import numpy as np
import imutils
import cv2

# --- Configuration ---
IP_CAM_URL = "http://192.168.0.4:25236/video" 

# How many frames to select. Fewer frames = faster stitching.
NUM_FRAMES_TO_STITCH = 20 

# --- OPTIMIZATION 1 ---
# Set a width for the images *used for stitching*.
# Smaller = MUCH faster. Larger = better quality. 
# 1000 is a good balance. 600 will be even faster.
STITCH_RESOLUTION_WIDTH = 1000
# ---

# --- Initialization ---
print(f"[INFO] connecting to IP camera at {IP_CAM_URL}...")
cap = cv2.VideoCapture(IP_CAM_URL)

if not cap.isOpened():
    print(f"[ERROR] Could not open video stream at {IP_CAM_URL}")
    exit()

print("\n" + "="*30)
print(" 🎥 IP Camera Stitcher (Optimized)")
print(" Controls:")
print("   's' - START capturing frames")
print("   'e' - END capturing and STITCH")
print("   'q' - QUIT")
print("="*30 + "\n")

captured_frames = [] 
is_capturing = False 

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from stream. Exiting.")
        break

    # Resize for *display* only
    display_frame = imutils.resize(frame, width=800)

    # --- Draw status information ---
    status_text = "STATUS: IDLE"
    color = (0, 255, 0) # Green

    if is_capturing:
        status_text = f"STATUS: CAPTURING ({len(captured_frames)} frames)"
        color = (0, 0, 255) # Red
        
        # --- OPTIMIZATION 1 (Implementation) ---
        # Resize the frame *before* appending it to save memory
        # and make stitching much faster.
        resized_for_stitching = imutils.resize(frame, width=STITCH_RESOLUTION_WIDTH)
        captured_frames.append(resized_for_stitching)
        # ---
    
    cv2.putText(display_frame, status_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(display_frame, "Press 's' to start, 'e' to stop/stitch, 'q' to quit", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Live IP Cam Feed (Press 's', 'e', 'q')", display_frame)

    key = cv2.waitKey(1) & 0xFF

    # --- 's' key: START ---
    if key == ord('s'):
        if not is_capturing:
            is_capturing = True
            captured_frames = []
            print("[INFO] ==> STARTING capture. Pan the camera slowly. Press 'e' to stop.")

    # --- 'e' key: END and STITCH ---
    elif key == ord('e'):
        if is_capturing:
            is_capturing = False
            print(f"[INFO] ==> ENDING capture. Total frames captured: {len(captured_frames)}")

            if len(captured_frames) < 2:
                print("[ERROR] Not enough frames to stitch.")
                continue

            # --- Frame Selection Logic (with trim) ---
            print(f"[INFO] Selecting ~{NUM_FRAMES_TO_STITCH} frames for stitching...")
            total = len(captured_frames)
            trim_percent = 0.1 # 10%
            start_index = int(total * trim_percent)
            end_index = int(total * (1.0 - trim_percent))

            if end_index <= start_index or (end_index - start_index) < 2:
                print("[ERROR] Not enough frames captured in the 'middle' segment. Try capturing for longer.")
                continue

            print(f"[INFO] Using frames from index {start_index} to {end_index}...")
            indices = np.linspace(start_index, end_index, num=NUM_FRAMES_TO_STITCH, dtype=int)
            indices = sorted(list(set(indices)))
            
            if len(indices) < 2:
                print(f"[ERROR] Only {len(indices)} unique frame(s) selected. Not enough to stitch.")
                continue

            # Our list of *resized* images to stitch
            images_to_stitch = [captured_frames[i] for i in indices]
            print(f"[INFO] Using {len(images_to_stitch)} unique frames (resolution: {images_to_stitch[0].shape[1]}w)")

            # --- Stitching Logic ---
            print("[INFO] Stitching images...")
            
            # --- OPTIMIZATION 3 ---
            # Try to set the stitching mode to "scan" which is faster for
            # simple left-to-right panning.
            try:
                stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
            except AttributeError:
                stitcher = cv2.createStitcher(cv2.Stitcher_SCANS)

            if stitcher is None:
                print("[INFO] SCANS mode not available, falling back to default PANORAMA mode.")
                stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
            # ---

            (status, stitched) = stitcher.stitch(images_to_stitch)

            # --- Handle Result ---
            if status == 0:
                print("[SUCCESS] Stitching successful!")
                
                # Display the output stitched image
                # We resize the *final* panorama only for display purposes
                stitched_display = imutils.resize(stitched, width=1200)
                cv2.imshow("Stitched Result", stitched_display)
                print("[INFO] Displaying stitched result. Press any key in the 'Stitched Result' window to close it.")
                cv2.waitKey(0)
                cv2.destroyWindow("Stitched Result")

            else:
                print(f"[ERROR] Image stitching failed (status={status})")
                print("  Tips: Try moving the camera slower and ensure 30-50% overlap.")

    # --- 'q' key: QUIT ---
    elif key == ord('q'):
        print("[INFO] Quitting program.")
        break

# --- Cleanup ---
print("[INFO] Releasing camera and closing windows.")
cap.release()
cv2.destroyAllWindows()
