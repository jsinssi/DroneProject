import cv2
from picamera2 import Picamera2
from roboflow import Roboflow

# --- Roboflow setup ---
rf = Roboflow(api_key="RUOJDNiiVKtE5T8Kro2x")
project = rf.workspace().project("parking-fhp6j-0ugk6")
model = project.version("1").model

# --- Pi Camera setup using Picamera2 ---
picam2 = Picamera2()

# Use RGB888 so colors are correct and consistent
camera_config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(camera_config)
picam2.start()

print("Camera started. Press 'q' in the window to quit.")

try:
    while True:
        # Capture a frame from the Pi camera as RGB
        frame_rgb = picam2.capture_array()  # This is already RGB (matches "RGB888")

        # Run Roboflow prediction on the RGB frame
        results = model.predict(frame_rgb, confidence=40).json()

        # We'll draw on a copy so we don't mutate the original too much
        frame_draw = frame_rgb.copy()

        # Draw bounding boxes
        for prediction in results.get("predictions", []):
            x = int(prediction["x"])
            y = int(prediction["y"])
            width = int(prediction["width"])
            height = int(prediction["height"])
            confidence = prediction["confidence"]
            label = prediction["class"]

            # Calculate box corners from center x, y, width, height
            x1 = x - width // 2
            y1 = y - height // 2
            x2 = x + width // 2
            y2 = y + height // 2

            # Draw rectangle and label
            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_draw,
                f"{label} ({confidence:.2f})",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # OpenCV expects BGR for display, so convert RGB → BGR
        frame_bgr = cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR)

        # Show the frame in a window on the Pi's monitor
        cv2.imshow("Roboflow Live Inference (Pi Camera)", frame_bgr)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera stopped, windows closed.")
