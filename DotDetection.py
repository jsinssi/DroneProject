import cv2
import numpy as np
from picamera2 import Picamera2

# --- GLOBAL INITIALISATION AND CONSTANTS ---
orb = cv2.ORB_create(nfeatures=500) 
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

previous_frame_gray = None
previous_keypoints = None
previous_descriptors = None

dot_map_locations = [] 
DOT_COUNT = 0
UNIQUENESS_THRESHOLD = 30 
CONNECTION_THRESHOLD = 100 
H_global = np.eye(3, dtype=np.float32)

# Global definitions for window control
LIVE_WINDOW_NAME = "Raspberry Pi Camera - Dot Counter (SLAM Mapping)"
MAP_WINDOW_NAME = "Final Dot Map Mosaic"
EXIT_FLAG = False

# --- NEW: Live Line Projection Function ---

def project_global_lines_to_frame(H_global_inv, dot_map_locations, connection_threshold):
    """
    Identifies connected dots in the global map and projects those line segments 
    back into the current frame's perspective.
    """
    lines_to_draw = []
    mapped_dots = np.array(dot_map_locations).astype(np.float32)
    
    if len(mapped_dots) < 2:
        return lines_to_draw

    # 1. Identify connected pairs in the global map
    for i in range(len(mapped_dots)):
        dot_A = mapped_dots[i]
        for j in range(i + 1, len(mapped_dots)):
            dot_B = mapped_dots[j]
            
            # Check distance in stable map coordinates
            distance = np.linalg.norm(dot_A - dot_B)
            
            if distance < connection_threshold:
                # Add the pair of global coordinates to the list
                lines_to_draw.append([dot_A, dot_B]) 

    if not lines_to_draw:
        return lines_to_draw

    # 2. Prepare points for projection
    all_global_pts = np.float32([pt for pair in lines_to_draw for pt in pair]).reshape(-1, 1, 2)
    
    # 3. Apply inverse Homography: Global Map -> Current Frame
    all_current_pts = cv2.perspectiveTransform(all_global_pts, H_global_inv).reshape(-1, 2)

    # 4. Format the projected points for drawing
    projected_lines = []
    for i in range(0, len(all_current_pts), 2):
        pt_A = tuple(all_current_pts[i].astype(int))
        pt_B = tuple(all_current_pts[i+1].astype(int))
        projected_lines.append((pt_A, pt_B))
        
    return projected_lines


# --- Mouse Callback Function (Unchanged) ---
def mouse_callback(event, x, y, flags, param):
    global EXIT_FLAG
    if event == cv2.EVENT_LBUTTONDOWN:
        h, w, _ = param.shape
        button_x1, button_y1, button_x2, button_y2 = w - 120, 20, w - 20, 60
        if button_x1 <= x <= button_x2 and button_y1 <= y <= button_y2:
            EXIT_FLAG = True

# --- DOT DETECTION FUNCTION (Unchanged) ---
def detect_dots(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range = np.array([0, 0, 0])
    upper_range = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dot_centers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10 and area < 500: 
            (x, y), radius = cv2.minEnclosingCircle(contour)
            dot_centers.append((int(x), int(y)))
    return frame, dot_centers

# --- FEATURE TRACKING (LOCALIZATION) FUNCTION (Unchanged) ---
def detect_and_track_features(current_frame, previous_frame_gray, previous_keypoints, previous_descriptors):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_keypoints, current_descriptors = orb.detectAndCompute(current_frame_gray, None)
    H = None 
    if previous_descriptors is not None and current_descriptors is not None and len(current_descriptors) > 4 and len(previous_descriptors) > 4:
        matches = bf.match(current_descriptors, previous_descriptors) 
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:100]
        if len(good_matches) > 4:
            src_pts = np.float32([current_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([previous_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return current_frame_gray, current_keypoints, current_descriptors, H

# --- DOT MAP UPDATE (MAPPING & COUNTING) FUNCTION (Unchanged) ---
def update_dot_map(dot_locations_current_frame, H_transform):
    global dot_map_locations, DOT_COUNT, UNIQUENESS_THRESHOLD
    if not dot_locations_current_frame: return
    dot_pts_current = np.float32(dot_locations_current_frame).reshape(-1, 1, 2)
    dot_pts_mapped = cv2.perspectiveTransform(dot_pts_current, H_transform).reshape(-1, 2)
    newly_mapped_dots = dot_pts_mapped.reshape(-1, 2)
    
    for mapped_dot in newly_mapped_dots:
        is_new_dot = True
        best_match_index = -1
        min_distance = UNIQUENESS_THRESHOLD
        
        for i, existing_dot in enumerate(dot_map_locations):
            distance = np.linalg.norm(mapped_dot - existing_dot)
            if distance < min_distance:
                min_distance = distance
                best_match_index = i
                is_new_dot = False
        
        if not is_new_dot and best_match_index != -1:
            existing_dot = dot_map_locations[best_match_index]
            alpha = 0.9 
            dot_map_locations[best_match_index] = (alpha * existing_dot) + ((1 - alpha) * mapped_dot)
        else:
            dot_map_locations.append(mapped_dot)
            DOT_COUNT += 1
            print(f"** NEW DOT DETECTED! Total Count: {DOT_COUNT} **")

# --- Dot Classification Helper Function (Unchanged) ---
def classify_dots_for_drawing(dot_locations_current_frame, H_transform, dot_map_locations, uniqueness_threshold):
    known_dots = []
    new_dots = []
    if not dot_locations_current_frame or len(dot_map_locations) == 0:
        return known_dots, dot_locations_current_frame 
    dot_pts_current = np.float32(dot_locations_current_frame).reshape(-1, 1, 2)
    dot_pts_mapped = cv2.perspectiveTransform(dot_pts_current, H_transform).reshape(-1, 2)
    for i, mapped_dot in enumerate(dot_pts_mapped):
        is_known = False
        for existing_dot in dot_map_locations:
            distance = np.linalg.norm(mapped_dot - existing_dot)
            if distance < uniqueness_threshold:
                is_known = True
                break
        original_center = dot_locations_current_frame[i] 
        if is_known:
            known_dots.append(original_center)
        else:
            new_dots.append(original_center)
    return known_dots, new_dots

# --- MAP VISUALIZATION FUNCTION (Unchanged) ---
def draw_dot_map(dot_locations):
    global EXIT_FLAG, DOT_COUNT, CONNECTION_THRESHOLD

    if not dot_locations:
        print("No dots were successfully mapped to visualize.")
        return

    mapped_dots = np.array(dot_locations).astype(np.float32)
    min_x, min_y = np.min(mapped_dots, axis=0).astype(int)
    max_x, max_y = np.max(mapped_dots, axis=0).astype(int)

    padding = 50 
    map_width = max_x - min_x + 2 * padding
    map_height = max_y - min_y + 2 * padding
    
    dot_map_image = np.full((map_height, map_width, 3), 255, dtype=np.uint8)

    x_offset = padding - min_x
    y_offset = padding - min_y

    print(f"\nDrawing final map: Size {map_width}x{map_height}, Total Dots: {len(mapped_dots)}")

    # 1. Draw connection lines
    for i in range(len(mapped_dots)):
        dot_A = mapped_dots[i]
        center_A = (int(dot_A[0] + x_offset), int(dot_A[1] + y_offset))
        
        for j in range(i + 1, len(mapped_dots)):
            dot_B = mapped_dots[j]
            distance = np.linalg.norm(dot_A - dot_B)
            
            if distance < CONNECTION_THRESHOLD:
                center_B = (int(dot_B[0] + x_offset), int(dot_B[1] + y_offset))
                cv2.line(dot_map_image, center_A, center_B, (150, 150, 150), 1)

    # 2. Draw the dots
    for dot in mapped_dots:
        center_x = int(dot[0] + x_offset)
        center_y = int(dot[1] + y_offset)
        cv2.circle(dot_map_image, (center_x, center_y), 4, (0, 0, 0), -1) 
        
    cv2.namedWindow(MAP_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(MAP_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cv2.setMouseCallback(MAP_WINDOW_NAME, mouse_callback, dot_map_image)

    while not EXIT_FLAG:
        map_display = dot_map_image.copy()

        # --- Draw UI elements ---
        h, w, _ = map_display.shape
        button_x1, button_y1, button_x2, button_y2 = w - 120, 20, w - 20, 60
        cv2.rectangle(map_display, (button_x1, button_y1), (button_x2, button_y2), (0, 0, 255), -1)
        cv2.putText(map_display, "EXIT", (button_x1 + 15, button_y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        count_text = f"Total Unique Dots: {DOT_COUNT}"
        cv2.putText(map_display, count_text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
        
        cv2.imshow(MAP_WINDOW_NAME, map_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Exiting program.")
    cv2.destroyAllWindows()


# --- MAIN LOOP (Updated for Live Lines) ---

def main_slam_mosaicing():
    global previous_frame_gray, previous_keypoints, previous_descriptors, H_global, EXIT_FLAG, UNIQUENESS_THRESHOLD, CONNECTION_THRESHOLD
    
    # --- Camera Setup ---
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": (320, 240)}) 
    picam2.configure(video_config)
    picam2.start()
    print("Press 'q' or click the red EXIT button to quit the video stream.")
    
    # --- Fullscreen Setup and Mouse Callback for Live Window ---
    cv2.namedWindow(LIVE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(LIVE_WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while not EXIT_FLAG: 
            current_frame = picam2.capture_array()
            display_frame = current_frame.copy()

            # --- Drawing preparation ---
            h, w, _ = display_frame.shape
            button_x1, button_y1, button_x2, button_y2 = w - 120, 20, w - 20, 60
            
            cv2.setMouseCallback(LIVE_WINDOW_NAME, mouse_callback, display_frame)

            # 1. Detect Dots and Track Features
            _, dot_locations_current = detect_dots(current_frame) 
            current_frame_gray, current_keypoints, current_descriptors, H_frame_to_previous = \
                detect_and_track_features(current_frame, previous_frame_gray, previous_keypoints, previous_descriptors)
            
            # --- LOCALIZATION AND MAPPING ---
            if H_frame_to_previous is not None:
                H_global = H_frame_to_previous @ H_global
                
                # Update the map (counting and refinement)
                update_dot_map(dot_locations_current, H_global)
                
                # --- NEW: Live Line Drawing Logic ---
                # Check for successful inversion (needed for projection back to current frame)
                try:
                    H_global_inv = np.linalg.inv(H_global)
                    
                    # Get projected line segments (in current frame coordinates)
                    projected_lines = project_global_lines_to_frame(H_global_inv, dot_map_locations, CONNECTION_THRESHOLD)
                    
                    # Draw the stable lines (GRAY)
                    for pt_A, pt_B in projected_lines:
                        # Draw thin gray line (BGR: 150, 150, 150)
                        cv2.line(display_frame, pt_A, pt_B, (150, 150, 150), 1)

                except np.linalg.LinAlgError:
                    # This happens if H_global is singular (e.g., in the very first few unstable frames)
                    pass

                # --- Classify and Draw Dots ---
                known_dots, new_dots = classify_dots_for_drawing(dot_locations_current, 
                                                                 H_global, 
                                                                 dot_map_locations, 
                                                                 UNIQUENESS_THRESHOLD)
                
                # Draw Known Dots in GREEN (BGR: 0, 255, 0) - Stable
                for center in known_dots:
                     cv2.circle(display_frame, center, 5, (0, 255, 0), -1)
                     
                # Draw New Dots in RED (BGR: 0, 0, 255) - New Data
                for center in new_dots:
                     cv2.circle(display_frame, center, 5, (0, 0, 255), -1) 

                # Display count on live feed
                cv2.putText(display_frame, f"Unique Dots: {DOT_COUNT}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # --- Draw UI elements on top ---
            cv2.rectangle(display_frame, (button_x1, button_y1), (button_x2, button_y2), (0, 0, 255), -1)
            cv2.putText(display_frame, "EXIT", (button_x1 + 15, button_y2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                 
            cv2.imshow(LIVE_WINDOW_NAME, display_frame)
            
            # 6. Update Previous Frame Data...
            previous_frame_gray = current_frame_gray
            previous_keypoints = current_keypoints
            previous_descriptors = current_descriptors
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nExiting video stream...")
        
    picam2.stop()
    cv2.destroyAllWindows()
    
    print("\n--- Final Dot Count and Estimated Locations ---")
    print(f"Total Unique Dots Counted: {DOT_COUNT}")
    
    # Final Map Visualization
    EXIT_FLAG = False 
    draw_dot_map(dot_map_locations)

if __name__ == "__main__":
    main_slam_mosaicing()
