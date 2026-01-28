# 1. Import the InferencePipeline library
from inference import InferencePipeline
import cv2

def my_sink(result, video_frame):
    if result.get("output_image"):  # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)
    
    # Overlay counted spaces on the video frame
    if "counted_spaces" in result:
        cv2.putText(video_frame, f"Counted Spaces: {result['counted_spaces']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the video frame with counted spaces
    cv2.imshow("Live Video Feed", video_frame)
    cv2.waitKey(1)

# 2. Initialize a pipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="RUOJDNiiVKtE5T8Kro2x",
    workspace_name="parking-project-kwqj7",
    workflow_id="detect-count-and-visualize-4",
    video_reference="http://100.113.43.83:8080/video",  # IP webcam stream URL
    max_fps=30,
    on_prediction=my_sink
)

# 3. Start the pipeline and wait for it to finish
pipeline.start()
pipeline.join()
