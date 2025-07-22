import cv2
import numpy as np
from PIL import Image
import io
from PIL import ImageDraw
from deskew import determine_skew

from ultralytics import YOLO  # pip install ultralytics

model = YOLO("./models/yolov8l.pt")

def detect_skew_angle(image_np):
    """
    Estimate skew angle of text in the image using Hough lines on edges.
    Returns angle in degrees. Positive means rotate CCW to correct.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    # Binarize image (adaptive threshold)
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)

    # Detect edges
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return 0.0  # no lines found, assume no skew

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle_deg = (theta * 180 / np.pi) - 90  # convert rad to degrees, rotate reference
        # We want angles close to 0, filter others
        if -45 < angle_deg < 45:
            angles.append(angle_deg)

    if len(angles) == 0:
        return 0.0

    # Average angle of detected lines
    median_angle = np.median(angles)
    return median_angle

def deskew_by_text(raw_bytes, debug=False):
    img = Image.open(io.BytesIO(raw_bytes)).convert("L")  # grayscale for skew detection
    img_np = np.array(img)

    angle = determine_skew(img_np)
    print(f"Detected text skew angle: {angle:.2f} degrees")

    (h, w) = img_np.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        img_np, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Convert rotated grayscale to 3-channel RGB
    rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_GRAY2RGB)

    if debug:
        Image.fromarray(rotated_rgb).save("deskewed_debug.png")

    return rotated_rgb


def crop_and_remove_code(raw_bytes, margin_percent=0.03, fill_color=(255,255,255)):
    # Deskew image first
    img_np = deskew_by_text(raw_bytes)

    h, w = img_np.shape[:2]

    results = model(img_np)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    indices = np.where(classes == 0)[0]
    if len(indices) == 0:
        print("No person detected in image.")
        return None

    areas = [(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]) for i in indices]
    i_largest = indices[np.argmax(areas)]
    x1, y1, x2, y2 = boxes[i_largest].astype(int)

    margin_px = int(w * margin_percent)
    crop_start_x = max(x2 - margin_px, 0)

    distance_pixels = w - x2
    percent_from_right = distance_pixels / w

    # Crop horizontally first (from crop_start_x to right edge)
    cropped_np = img_np[:, crop_start_x:]
    cropped_img = Image.fromarray(cropped_np)

    # Person box coords relative to cropped image
    person_x1 = x1 - crop_start_x
    person_x2 = x2 - crop_start_x
    person_y1 = y1
    person_y2 = y2

    photo_height = person_y2 - person_y1

    # Erase rectangle parameters (tuned):
    erase_left = person_x1 + int(photo_height * 2.2)  # start 2.2× photo height right of photo left edge
    erase_right = erase_left + int(w * 0.15)          # width ~15% of image width

    erase_top = person_y1 + int(photo_height * 0.13)   # start 0.13x of photo top
    erase_bottom = erase_top + int(photo_height * 0.25)  # cover top 25% of photo height

    # Clamp erase box within cropped image boundaries
    erase_left = max(erase_left, 0)
    erase_right = min(erase_right, cropped_img.width)
    erase_top = max(erase_top, 0)
    erase_bottom = min(erase_bottom, cropped_img.height)

    if erase_right > erase_left and erase_bottom > erase_top:
        draw = ImageDraw.Draw(cropped_img)
        draw.rectangle([erase_left, erase_top, erase_right, erase_bottom], fill=fill_color)
    else:
        print("Erase box invalid or zero area, skipping erase")

    # --- New crop to remove bottom starting from y2 + 30% photo height ---
    crop_start_y = y2 + int(photo_height * 0.3)
    crop_start_y = min(crop_start_y, h)  # Clamp to image height

    # Crop the image vertically, remove from crop_start_y to bottom
    # Adjust crop_start_y relative to cropped image top (which is 0 vertically)
    # Since we cropped horizontally only, vertical indices remain the same.
    cropped_img = cropped_img.crop((0, 0, cropped_img.width, crop_start_y))

    return cropped_img, [x1, y1, x2, y2], distance_pixels, percent_from_right


