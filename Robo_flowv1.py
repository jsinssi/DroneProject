# 1. Import the InferencePipeline library
from inference import InferencePipeline
import cv2
import numpy as np

# Create resizable window for workflow image
cv2.namedWindow("Workflow Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Workflow Image", 800, 600)

def my_sink(result, video_frame):
    if result.get("output_image"):
        try:
            output_image = result["output_image"]
            if hasattr(output_image, 'numpy_image'):
                output_np = output_image.numpy_image
            elif hasattr(output_image, 'image'):
                output_np = output_image.image
            else:
                output_np = np.asarray(output_image)
            
            # Ensure array is writable for OpenCV text overlay
            output_np = np.ascontiguousarray(output_np)
            
            # Calculate counts from predictions
            counts = {}
            if "predictions" in result:
                for pred in result["predictions"]:
                    # Handle both dictionary and object access
                    class_name = pred.get("class") if isinstance(pred, dict) else getattr(pred, "class_name", "unknown")
                    counts[class_name] = counts.get(class_name, 0) + 1
            
            # Also check for explicit aggregation output 'count_objects' from workflow
            elif "count_objects" in result:
                # If it's a number, display it as "Total"
                if isinstance(result["count_objects"], (int, float)):
                     counts["Total"] = result["count_objects"]
                elif isinstance(result["count_objects"], dict):
                     counts.update(result["count_objects"])

            y_pos = 30
            # Display all found counts
            for label, count in counts.items():
                # Choose color based on label content
                color = (0, 255, 0) # Green default
                if "occupied" in str(label).lower() or "used" in str(label).lower():
                    color = (0, 0, 255) # Red for occupied
                
                text = f"{label}: {count}"
                cv2.putText(output_np, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                y_pos += 40
            
            # Fallback if no counts found
            if not counts:
                 # Debug: print keys to help user
                 # print(f"Available keys: {list(result.keys())}")
                 pass
            
            cv2.imshow("Workflow Image", output_np)
            cv2.waitKey(1)
        except Exception as e:
            print(f"Error displaying output: {e}")

# 2. Initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="RUOJDNiiVKtE5T8Kro2x",
    workspace_name="parking-project-kwqj7",
    workflow_id="detect-count-and-visualize-4",
    video_reference="http://100.64.135.100:8080/video",
    max_fps=30,
    on_prediction=my_sink
)

# 3. Start the pipeline and wait for it to finish
pipeline.start()
pipeline.join()
