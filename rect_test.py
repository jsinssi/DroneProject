import cv2
import numpy as np
from picamera2 import Picamera2


# -------------------------
# Landmark detection (same as before)
# -------------------------

def find_colored_circle(hsv, color_name):
    """
    Find the centroid of a coloured landmark circle.
    Returns (cx, cy) or None.
    """

    if color_name == "red":
        # Red wraps around 0°, so we use two ranges
        lower1 = np.array([0, 120, 70])
        upper1 = np.array([10, 255, 255])
        lower2 = np.array([170, 120, 70])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)

    elif color_name == "green":
        lower = np.array([35, 80, 80])
        upper = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    elif color_name == "blue":
        # Tighter, saturated blue range to avoid table / paper
        lower = np.array([100, 100, 40])   # H, S, V
        upper = np.array([135, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)

    else:
        return None

    # Clean the mask (remove specks)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Filter contours by area and "circular-ish" shape
    MIN_AREA = 500
    MAX_AREA = 20000
    best_cnt = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)
        if not (0.7 <= aspect <= 1.3):
            continue  # not roughly square/circular

        if area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None:
        return None

    M = cv2.moments(best_cnt)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


# -------------------------
# Rectangle detection (new)
# -------------------------

def detect_rectangles(gray):
    """
    Detect black rectangles on a light background.
    Returns list of (cx, cy) centroids.
    """
    h, w = gray.shape[:2]

    # Slight blur, then Otsu threshold (dark rectangles on light paper)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rect_centres = []

    MIN_AREA = 800      # tune if needed
    MAX_AREA = 60000
    MARGIN = 10         # ignore things touching image border
    MIN_SIZE = 20

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA or area > MAX_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        # Ignore stuff touching the very edge (table, shadows, etc.)
        if x < MARGIN or y < MARGIN or (x + bw) > (w - MARGIN) or (y + bh) > (h - MARGIN):
            continue

        if bw < MIN_SIZE or bh < MIN_SIZE:
            continue

        aspect = bw / float(bh)
        if not (0.7 <= aspect <= 1.3):    # rectangles are roughly square
            continue

        # Approx polygon to be sure it is 4-sided
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        rect_centres.append((cx, cy))

    return rect_centres, th


# -------------------------
# Camera setup
# -------------------------

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

print("Landmarks + rectangle detection running. Press 'q' to quit.")

try:
    while True:
        frame = picam2.capture_array()
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1. Detect landmarks
        A = find_colored_circle(hsv, "red")
        B = find_colored_circle(hsv, "green")
        C = find_colored_circle(hsv, "blue")

        # Draw A/B/C
        if A is not None:
            x, y = A
            cv2.circle(bgr, (x, y), 20, (0, 0, 255), 3)
            cv2.putText(bgr, "A", (x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if B is not None:
            x, y = B
            cv2.circle(bgr, (x, y), 20, (0, 255, 0), 3)
            cv2.putText(bgr, "B", (x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if C is not None:
            x, y = C
            cv2.circle(bgr, (x, y), 20, (255, 0, 0), 3)
            cv2.putText(bgr, "C", (x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 2. Detect rectangles
        rect_centres, th = detect_rectangles(gray)

        # Draw rectangle centres
        for (cx, cy) in rect_centres:
            cv2.circle(bgr, (cx, cy), 6, (255, 0, 255), 2)  # magenta dot

        # Show count on screen + print to terminal
        count = len(rect_centres)
        cv2.putText(bgr, f"Rects this frame: {count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        print("Rects this frame:", count)

        # Show views
        cv2.imshow("Landmarks + Rectangles", bgr)
        # Optional debug: threshold view
        # cv2.imshow("Threshold", th)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Done.")

