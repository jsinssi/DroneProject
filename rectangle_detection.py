import cv2
import numpy as np

def find_black_rectangles(frame, debug=False):
    """
    Detects black rectangles on a white-ish background.
    Returns:
      rect_centers: list of (x, y) centers in image pixels
      rect_boxes:   list of (x, y, w, h) bounding boxes
      vis_frame:    frame with drawings if debug=True, else original
    """
    # 1. Grayscale + blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. Threshold: black -> white (foreground), white page -> black (background)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 3. Small morphological open to clean specks
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # 4. Find external contours (rectangles should appear as blobs)
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rect_centers = []
    rect_boxes = []

    vis = frame.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:       # ignore tiny noise; tune for your size
            continue

        # Approximate contour to polygon
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # We want convex 4-sided polygons
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)

            # Ignore very skinny / tiny shapes
            if w < 10 or h < 10:
                continue

            aspect = w / float(h)
            # Optional: restrict aspect ratio range if your rectangles are roughly square
            if aspect < 0.3 or aspect > 3.5:
                continue

            # Check how "full" the rectangle is (to skip frames with weird noise)
            rect_area = w * h
            fill_ratio = area / float(rect_area)
            if fill_ratio < 0.5:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0

            rect_centers.append((cx, cy))
            rect_boxes.append((x, y, w, h))

            if debug:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(vis, (int(cx), int(cy)), 4, (0, 255, 0), -1)

    return rect_centers, rect_boxes, (vis if debug else frame)
