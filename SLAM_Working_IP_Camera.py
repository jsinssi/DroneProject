import cv2
import numpy as np
import math

# ================= CONFIG =================

# ArUco marker centres in mm (origin: bottom-left of A4 in landscape)
MARKER_WORLD_MM = {
    1: (186.0, 278.0),
    3: (188.0, 187.0),
    5: (189.0, 95.0),
    7: (189.0, 34.0),

    0: (27.0, 283.0),
    2: (23.0, 186.0),
    4: (23.0, 97.0),
    6: (25.0, 35.0),
}

# Bay centres in A4 coordinates (mm)
BAY_WORLD_MM = {
    0: (172.0, 247.0),
    1: (172.0, 213.0),
    2: (172.0, 157.0),
    3: (172.0, 126.0),
    4: (172.0, 65.0),

    5: (45.0, 247.0),
    6: (45.0, 214.0),
    7: (45.0, 157.0),
    8: (45.0, 126.0),
    9: (50.0, 67.0),
}

# Dimensions of a parking bay in mm
BAY_WIDTH_MM = 30.0
BAY_HEIGHT_MM = 50.0

# HSV color ranges for car detection (yellow and orange)
# To find the range for your car, you can use an online HSV color picker.
CAR_COLOR_RANGES = [
    # Orange
    {'lower': np.array([5, 100, 100]), 'upper': np.array([20, 255, 255])},
    # Yellow
    {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])}
]
OCCUPANCY_THRESHOLD = 0.3  # 30% of pixels must be car color to be 'occupied'

MAX_SPACES = len(BAY_WORLD_MM)

ASSIGN_MAX_DIST_MM = 25.0
MIN_RECT_AREA = 1000  # Reduced min area to detect smaller cars
MAX_RECT_AREA = 80000
MIN_ASPECT = 0.3
#MAX_ASPECT = 3.5


# ================= KALMAN FILTER ON HOMOGRAPHY =================

class HomographyKalmanFilter:
    def __init__(self, process_var=1e-4, meas_var=1e-2):
        self.n = 9
        self.x = np.zeros((self.n, 1), dtype=np.float32)
        self.P = np.eye(self.n, dtype=np.float32) * 1e6
        self.F = np.eye(self.n, dtype=np.float32)
        self.Q = np.eye(self.n, dtype=np.float32) * process_var
        self.R = np.eye(self.n, dtype=np.float32) * meas_var
        self.initialised = False

    def predict(self):
        if not self.initialised:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, H_meas):
        h = H_meas.astype(np.float32).reshape(self.n, 1)

        if not self.initialised:
            self.x = h
            self.P = np.eye(self.n, dtype=np.float32)
            self.initialised = True
            return

        self.predict()

        I = np.eye(self.n, dtype=np.float32)
        z = h
        y = z - self.x
        S = self.P + self.R
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (I - K) @ self.P

    def get_H(self):
        if not self.initialised:
            return None
        H = self.x.reshape(3, 3)
        denom = H[2, 2]
        if abs(denom) > 1e-6:
            H = H / denom
        return H


# ================= DETECTION + GEOMETRY HELPERS =================

def compute_homography_from_aruco(gray):
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, corners, ids

    ids = ids.flatten()
    img_pts, world_pts = [], []
    for i, marker_id in enumerate(ids):
        if marker_id in MARKER_WORLD_MM:
            c = corners[i][0]
            center = c.mean(axis=0)
            img_pts.append(center)
            world_pts.append(MARKER_WORLD_MM[marker_id])

    if len(img_pts) < 4:
        return None, corners, ids

    img_pts = np.array(img_pts, dtype=np.float32)
    world_pts = np.array(world_pts, dtype=np.float32)
    H_raw, _ = cv2.findHomography(img_pts, world_pts, method=0)
    return H_raw, corners, ids


def image_to_world(H, pixel_point):
    pts = np.array([[pixel_point]], dtype=np.float32)
    world = cv2.perspectiveTransform(pts, H)
    return world[0, 0]

def world_to_image(H_inv, world_point):
    pts = np.array([[world_point]], dtype=np.float32)
    img = cv2.perspectiveTransform(pts, H_inv)
    return img[0, 0]

def detect_car_rectangles(hsv_image, marker_boxes):
    """Detects cars based on color and returns their bounding boxes."""
    # Create a combined color mask for all specified car colors
    combined_color_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for color_range in CAR_COLOR_RANGES:
        color_mask = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
        combined_color_mask = cv2.bitwise_or(combined_color_mask, color_mask)

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_RECT_AREA or area > MAX_RECT_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Skip if the detected object is inside a marker box
        cx = x + w / 2.0
        cy = y + h / 2.0
        is_marker = False
        for (mx, my, mw, mh) in marker_boxes:
            if (mx <= cx <= mx + mw) and (my <= cy <= my + mh):
                is_marker = True
                break
        if is_marker:
            continue

        rects.append((x, y, w, h))

    return rects

def detect_parking_rectangles(gray, marker_boxes):
    """
    Detect parking rectangles using edges (Canny).
    Orientation-invariant: accepts the same rectangles whether the page
    is held portrait or landscape (no assumption h > w).
    Returns list of bounding boxes: [(x, y, w, h), ...] in IMAGE PIXELS.
    """
    h_img, w_img = gray.shape

    margin_x = int(0.1 * w_img)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_RECT_AREA or area > MAX_RECT_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Ignore things near left/right edges (likely markers / desk / borders)
        if x < margin_x or x + w > (w_img - margin_x):
            continue

        # Orientation-invariant aspect ratio:
        short_side = min(w, h)
        long_side = max(w, h)
        aspect = short_side / float(long_side)  # between 0 and 1

        # Require a reasonable rectangle shape (not a very thin line)
        if aspect < MIN_ASPECT or aspect > 1.0:  # MAX_ASPECT not needed now
            continue

        # Centre of this rectangle in image coords
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Skip if centre lies inside any marker box
        is_marker = False
        for (mx, my, mw, mh) in marker_boxes:
            if (mx <= cx <= mx + mw) and (my <= cy <= my + mh):
                is_marker = True
                break
        if is_marker:
            continue

        rects.append((x, y, w, h))

    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    return rects[:MAX_SPACES]


def assign_to_bay(wx, wy):
    best_id = None
    best_d2 = ASSIGN_MAX_DIST_MM ** 2

    for bay_id, (bx, by) in BAY_WORLD_MM.items():
        dx = wx - bx
        dy = wy - by
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_id = bay_id

    return best_id


# ================= MAIN LOOP =================

def main():
    cap = cv2.VideoCapture("http://100.121.61.223:25236/video")
    if not cap.isOpened():
        print("Could not open camera")
        return

    kf_H = HomographyKalmanFilter(process_var=1e-4, meas_var=1e-2)
    
    # Use a dictionary to store the status and rectangle for each bay
    bay_states = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1. Markers + raw homography (only if ≥4 markers)
        H_raw, marker_corners, marker_ids = compute_homography_from_aruco(gray)

        marker_boxes = []
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            for c in marker_corners:
                x_m, y_m, w_m, h_m = cv2.boundingRect(c[0].astype(np.int32))
                marker_boxes.append((x_m, y_m, w_m, h_m))

        # 2. Kalman update / predict
        if H_raw is not None:
            kf_H.update(H_raw)
        else:
            kf_H.predict()

        H_used = kf_H.get_H()
        if H_used is None:
            H_used = H_raw  # before KF initialises fully

        # 3. Detect empty bays and occupied bays (cars)
        empty_rects_img = detect_parking_rectangles(gray, marker_boxes)
        car_rects_img = detect_car_rectangles(hsv, marker_boxes)

        # 4. Determine status of all bays before drawing
        if H_used is not None:
            bay_states.clear()

            # First, mark all detected empty bays
            for (x, y, w, h) in empty_rects_img:
                cx = x + w / 2.0
                cy = y + h / 2.0
                wx, wy = image_to_world(H_used, (cx, cy))
                bay_id = assign_to_bay(wx, wy)
                if bay_id is not None:
                    bay_states[bay_id] = {'status': 'Empty', 'rect': (x, y, w, h)}

            # Then, mark all detected cars, overwriting empty status if necessary
            for (x, y, w, h) in car_rects_img:
                cx = x + w / 2.0
                cy = y + h / 2.0
                wx, wy = image_to_world(H_used, (cx, cy))
                bay_id = assign_to_bay(wx, wy)
                if bay_id is not None:
                    bay_states[bay_id] = {'status': 'Occupied', 'rect': (x, y, w, h)}

            # 5. Draw all bays based on their final state
            for bay_id, state in bay_states.items():
                x, y, w, h = state['rect']
                status = state['status']
                
                if status == 'Empty':
                    color = (0, 255, 0)  # Green
                else: # Occupied
                    color = (0, 0, 255)  # Red

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"Bay {bay_id}: {status}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 0), 2, cv2.LINE_AA)
        else:
            # Pose not initialised yet – just show raw detected rectangles
            for (x, y, w, h) in empty_rects_img:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (x, y, w, h) in car_rects_img:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # HUD
        occupied_count = sum(1 for s in bay_states.values() if s['status'] == 'Occupied')
        empty_count = sum(1 for s in bay_states.values() if s['status'] == 'Empty')

        cv2.putText(frame, f"Occupied bays: {occupied_count}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Empty bays: {empty_count}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Kalman-filtered map-based vSLAM parking demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
