# Use:
# python -m uvicorn main:app --reload
# in terminal after starting it
import cv2
import numpy as np
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import uvicorn
import sqlite3
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# ================= DATABASE CONFIG =================
DATABASE_FILE = "parking_history.db"
# Use a thread pool for synchronous database operations to avoid blocking the event loop
db_executor = ThreadPoolExecutor(max_workers=5)

def init_db():
    """Initializes the SQLite database and creates the history table if it doesn't exist."""
    # This is a quick, one-time operation, so running it synchronously at startup is acceptable.
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS occupancy_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bay_id INTEGER NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            duration_seconds INTEGER NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def _execute_add_history(bay_id, start_time_iso, end_time_iso, duration_seconds):
    """Synchronous function to be run in the executor."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO occupancy_history (bay_id, start_time, end_time, duration_seconds)
        VALUES (?, ?, ?, ?)
    """, (bay_id, start_time_iso, end_time_iso, duration_seconds))
    conn.commit()
    conn.close()

async def add_history_record(bay_id, start_time, end_time):
    """Adds a new occupancy record to the history database asynchronously."""
    duration = end_time - start_time
    if duration.total_seconds() < 3:
        return # Do not record if occupancy is less than 3 seconds

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        db_executor, 
        _execute_add_history, 
        bay_id, 
        start_time.isoformat(), 
        end_time.isoformat(), 
        int(duration.total_seconds())
    )

def _execute_get_history():
    """Synchronous function to fetch history, to be run in the executor."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM occupancy_history ORDER BY end_time DESC LIMIT 3")
    records = cursor.fetchall()
    conn.close()
    return [dict(row) for row in records]

# ================= CONFIG (from Slam_Working.py) =================

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
CAR_COLOR_RANGES = [
    # Orange
    {'lower': np.array([5, 100, 100]), 'upper': np.array([20, 255, 255])},
    # Yellow
    {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])}
]
OCCUPANCY_THRESHOLD = 0.3  # 30% of pixels must be car color to be 'occupied'

MAX_SPACES = len(BAY_WORLD_MM)

ASSIGN_MAX_DIST_MM = 25.0
MIN_RECT_AREA = 1000
MAX_RECT_AREA = 80000
MIN_ASPECT = 0.3

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
    combined_color_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)
    for color_range in CAR_COLOR_RANGES:
        color_mask = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
        combined_color_mask = cv2.bitwise_or(combined_color_mask, color_mask)

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
        if x < margin_x or x + w > (w_img - margin_x):
            continue
        short_side = min(w, h)
        long_side = max(w, h)
        aspect = short_side / float(long_side)
        if aspect < MIN_ASPECT or aspect > 1.0:
            continue
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

# ================= FASTAPI APPLICATION =================

app = FastAPI()

# Global state to store the latest processed frame and bay statuses
latest_frame = None
bay_states = {} # Current status for API
occupancy_tracker = {} # For tracking start times and history logic

async def video_processing_loop():
    """The main video processing logic, adapted to run as a background task."""
    global latest_frame, bay_states, occupancy_tracker
    
    cap = cv2.VideoCapture("http://10.76.95.116:25236/video")
    if not cap.isOpened():
        print("Could not open camera")
        return

    kf_H = HomographyKalmanFilter(process_var=1e-4, meas_var=1e-2)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.1) # Wait a bit before retrying
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        H_raw, marker_corners, marker_ids = compute_homography_from_aruco(gray)

        marker_boxes = []
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
            for c in marker_corners:
                x_m, y_m, w_m, h_m = cv2.boundingRect(c[0].astype(np.int32))
                marker_boxes.append((x_m, y_m, w_m, h_m))

        if H_raw is not None:
            kf_H.update(H_raw)
        else:
            kf_H.predict()

        H_used = kf_H.get_H()
        if H_used is None:
            H_used = H_raw

        empty_rects_img = detect_parking_rectangles(gray, marker_boxes)
        car_rects_img = detect_car_rectangles(hsv, marker_boxes)

        current_bay_states = {}
        if H_used is not None:
            # Process empty bays first
            for (x, y, w, h) in empty_rects_img:
                cx, cy = x + w / 2.0, y + h / 2.0
                wx, wy = image_to_world(H_used, (cx, cy))
                bay_id = assign_to_bay(wx, wy)
                if bay_id is not None:
                    current_bay_states[bay_id] = {'status': 'Empty', 'rect': (x, y, w, h)}

            # Process occupied bays, potentially overwriting empty status
            for (x, y, w, h) in car_rects_img:
                cx, cy = x + w / 2.0, y + h / 2.0
                wx, wy = image_to_world(H_used, (cx, cy))
                bay_id = assign_to_bay(wx, wy)
                if bay_id is not None:
                    current_bay_states[bay_id] = {'status': 'Occupied', 'rect': (x, y, w, h)}

            # History logic: Compare current states with the tracker
            now = datetime.now()
            
            # Check for newly occupied bays
            for bay_id, state in current_bay_states.items():
                if state['status'] == 'Occupied' and bay_id not in occupancy_tracker:
                    occupancy_tracker[bay_id] = {'status': 'Occupied', 'start_time': now}

            # Check for newly empty bays
            for bay_id, tracked_state in list(occupancy_tracker.items()):
                if bay_id not in current_bay_states or current_bay_states[bay_id]['status'] == 'Empty':
                    # Fire and forget the async database write
                    asyncio.create_task(add_history_record(bay_id, tracked_state['start_time'], now))
                    del occupancy_tracker[bay_id]

            # Drawing logic
            for bay_id, state in current_bay_states.items():
                x, y, w, h = state['rect']
                color = (0, 255, 0) if state['status'] == 'Empty' else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"Bay {bay_id}: {state['status']}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            # Fallback drawing if no homography
            for (x, y, w, h) in empty_rects_img:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for (x, y, w, h) in car_rects_img:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        occupied_count = sum(1 for s in current_bay_states.values() if s['status'] == 'Occupied')
        empty_count = len(BAY_WORLD_MM) - occupied_count

        cv2.putText(frame, f"Occupied: {occupied_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Empty: {empty_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Update global state for APIs
        bay_states = {k: {'status': v['status']} for k, v in current_bay_states.items()}
        _, encoded_frame = cv2.imencode('.jpg', frame)
        latest_frame = encoded_frame.tobytes()
        
        await asyncio.sleep(0.01) # Yield control to the event loop

async def generate_video_stream():
    """Yields the latest processed frame for the streaming response."""
    while True:
        if latest_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
        await asyncio.sleep(0.05) # Stream at ~20 FPS

@app.on_event("startup")
async def startup_event():
    """Initialize DB and start the video processing task on server startup."""
    init_db()
    asyncio.create_task(video_processing_loop())

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serves a modern, user-friendly HTML page with auto-updating counts and a clock."""
    # Initial counts for the first page load
    occupied_count = sum(1 for s in bay_states.values() if s['status'] == 'Occupied')
    total_bays = len(BAY_WORLD_MM)
    empty_count = total_bays - occupied_count
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>SkySpot SLAM Demo</title>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">
            <style>
                body {{
                    font-family: 'Roboto', sans-serif;
                    background-color: #f0f2f5;
                    color: #333;
                    margin: 0;
                    padding: 20px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                header {{
                    width: 100%;
                    max-width: 1200px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                header h1 {{
                    color: #1d3557;
                    margin: 0;
                }}
                #clock {{
                    font-size: 22px;
                    font-weight: 700;
                    color: #457b9d;
                    background-color: #fff;
                    padding: 10px 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                main {{
                    display: flex;
                    gap: 20px;
                    width: 100%;
                    max-width: 1200px;
                }}
                .sidebar {{
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                    flex-basis: 300px;
                    flex-shrink: 0;
                }}
                .status-container, .history-container {{
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .status-container h2, .history-container h2 {{
                    margin-top: 0;
                    color: #1d3557;
                    border-bottom: 2px solid #f1faee;
                    padding-bottom: 10px;
                }}
                .status-item {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 20px;
                    margin-bottom: 15px;
                }}
                .status-item strong {{
                    font-size: 28px;
                }}
                .occupied {{ color: #e63946; }}
                .empty {{ color: #2a9d8f; }}
                .video-container {{
                    flex-grow: 1;
                    background-color: #000;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .video-container img {{
                    width: 100%;
                    display: block;
                }}
                #history-list {{
                    list-style-type: none;
                    padding: 0;
                    margin: 0;
                    max-height: 300px;
                    overflow-y: auto;
                }}
                #history-list li {{
                    background-color: #f8f9fa;
                    border-left: 4px solid #457b9d;
                    padding: 8px 12px;
                    margin-bottom: 8px;
                    font-size: 14px;
                    border-radius: 4px;
                }}
                footer {{
                    margin-top: 20px;
                    color: #6c757d;
                }}
                footer a {{
                    color: #457b9d;
                    text-decoration: none;
                }}
                footer a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <header>
                <h1>SkySpot SLAM Demo</h1>
                <div id="clock"></div>
            </header>
            <main>
                <div class="sidebar">
                    <div class="status-container">
                        <h2>Live Status</h2>
                        <div class="status-item">
                            <span>Occupied Spaces:</span>
                            <strong id="occupied-count" class="occupied">{occupied_count}</strong>
                        </div>
                        <div class="status-item">
                            <span>Empty Spaces:</span>
                            <strong id="empty-count" class="empty">{empty_count}</strong>
                        </div>
                    </div>
                    <div class="history-container">
                        <h2>Occupancy History</h2>
                        <ul id="history-list">
                            <!-- History items will be injected here -->
                        </ul>
                    </div>
                </div>
                <div class="video-container">
                    <img src="/video" alt="Live video feed">
                </div>
            </main>
            <footer>
                <p>API for raw data available at <a href="/status" target="_blank">/status</a> and <a href="/history" target="_blank">/history</a>.</p>
            </footer>

            <script>
                const totalBays = {total_bays};
                const occupiedCountElement = document.getElementById('occupied-count');
                const emptyCountElement = document.getElementById('empty-count');
                const clockElement = document.getElementById('clock');
                const historyListElement = document.getElementById('history-list');

                async function updateCounts() {{
                    try {{
                        const response = await fetch('/status');
                        const data = await response.json();
                        
                        let occupiedCount = 0;
                        const allBays = new Set(Array.from({{length: totalBays}}, (_, i) => i));
                        
                        for (const bayId in data) {{
                            if (data[bayId].status === 'Occupied') {{
                                occupiedCount++;
                            }}
                        }}
                        
                        const emptyCount = totalBays - occupiedCount;

                        occupiedCountElement.textContent = occupiedCount;
                        emptyCountElement.textContent = emptyCount;
                    }} catch (error) {{
                        console.error("Error fetching status:", error);
                    }}
                }}

                async function updateHistory() {{
                    try {{
                        const response = await fetch('/history');
                        const historyData = await response.json();
                        
                        historyListElement.innerHTML = ''; // Clear current list

                        if (historyData.length === 0) {{
                            historyListElement.innerHTML = '<li>No recent history.</li>';
                            return;
                        }}

                        historyData.forEach(item => {{
                            const li = document.createElement('li');
                            const startTime = new Date(item.start_time);
                            const duration = item.duration_seconds;
                            const timeString = startTime.toLocaleTimeString('en-US', {{ hour: '2-digit', minute: '2-digit' }});
                            
                            let durationText = '';
                            if (duration < 60) {{
                                durationText = `${{duration}}s`;
                            }} else {{
                                const minutes = Math.floor(duration / 60);
                                const seconds = duration % 60;
                                durationText = `${{minutes}}m ${{seconds}}s`;
                            }}

                            li.textContent = `Bay ${{item.bay_id}} occupied at ${{timeString}} for ${{durationText}}.`;
                            historyListElement.appendChild(li);
                        }});
                    }} catch (error) {{
                        console.error("Error fetching history:", error);
                    }}
                }}

                function updateClock() {{
                    const now = new Date();
                    const timeString = now.toLocaleTimeString('en-US');
                    clockElement.textContent = timeString;
                }}

                setInterval(updateCounts, 2000);
                setInterval(updateHistory, 5000); // Update history every 5 seconds
                setInterval(updateClock, 1000);
                
                // Initial calls to populate data immediately
                updateClock();
                updateCounts();
                updateHistory();
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video")
async def video_feed():
    """Returns the processed video stream."""
    return StreamingResponse(generate_video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
async def get_status():
    """Returns the current status of all detected parking bays."""
    return bay_states

@app.get("/history")
async def get_history():
    """Returns the last 3 occupancy history records asynchronously."""
    loop = asyncio.get_running_loop()
    records = await loop.run_in_executor(db_executor, _execute_get_history)
    return JSONResponse(content=records)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
