import os
import cv2
import numpy as np
from inference import InferencePipeline

# --- Configuration ---
# 0 is the default index for CSI cameras (when using V4L2/Legacy drivers)
VIDEO_SOURCE = 0 
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "RUOJDNiiVKtE5T8Kro2x")
WORKSPACE_NAME = "parking-project-kwqj7"
WORKFLOW_ID = "detect-count-and-visualize-4"

# Check if a display is available. 
# If running via SSH without X11 forwarding, this will be False.
HEADLESS = os.environ.get("DISPLAY") is None

if not HEADLESS:
    cv2.namedWindow("Workflow Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Workflow Image", 800, 600)
else:
    print("Running in headless mode (no monitor detected). Visual output disabled; printing to console.")

def my_sink(result, video_frame):
    # Only process if we have an output image
    if result.get("output_image"):
        try:
            output_image = result["output_image"]
            
            # 1. Convert Inference image to a standard numpy array
            if hasattr(output_image, 'numpy_image'):
                output_np = output_image.numpy_image
            elif hasattr(output_image, 'image'):
                output_np = output_image.image
            else:
                output_np = np.asarray(output_image)
            
            # Ensure array is contiguous for OpenCV operations
            output_np = np.ascontiguousarray(output_np)
            
            # 2. Extract Counts from Predictions
            counts = {}
            
            # Strategy A: Aggregate from raw predictions list
            if "predictions" in result:
                for pred in result["predictions"]:
                    # Handle both dictionary and object access styles
                    class_name = pred.get("class") if isinstance(pred, dict) else getattr(pred, "class_name", "unknown")
                    counts[class_name] = counts.get(class_name, 0) + 1
            
            # Strategy B: Check for explicit aggregate outputs from workflow
            elif "count_objects" in result:
                if isinstance(result["count_objects"], (int, float)):
                      counts["Total"] = result["count_objects"]
                elif isinstance(result["count_objects"], dict):
                      counts.update(result["count_objects"])

            # 3. output Logic
            y_pos = 30
            
            # If we are headless, print to console to verify it's working
            if HEADLESS and counts:
                print(f"--- Frame Stats ---")
            
            for label, count in counts.items():
                if HEADLESS:
                    print(f"{label}: {count}")
                else:
                    # Choose color: Red for occupied, Green for others
                    color = (0, 255, 0) 
                    if "occupied" in str(label).lower() or "used" in str(label).lower():
                        color = (0, 0, 255) 
                    
                    text = f"{label}: {count}"
                    cv2.putText(output_np, text, (10, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    y_pos += 40

            # 4. Display Frame (Only if monitor exists)
            if not HEADLESS:
                cv2.imshow("Workflow Image", output_np)
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

        except Exception as e:
            print(f"Error processing frame: {e}")

# Initialize the pipeline with video_reference=0 for local camera
pipeline = InferencePipeline.init_with_workflow(
    api_key=API_KEY,
    workspace_name=WORKSPACE_NAME,
    workflow_id=WORKFLOW_ID,
    video_reference=VIDEO_SOURCE, 
    max_fps=30,
    on_prediction=my_sink
)

print("Starting pipeline... Press Ctrl+C to stop.")
try:
    pipeline.start()
    pipeline.join()
except KeyboardInterrupt:
    print("\nStopping pipeline...")
    pipeline.terminate()
    if not HEADLESS:
        cv2.destroyAllWindows()
