import cv2
import numpy as np
import time

# ---------------------------------------------------------
# Configurable Parameters
# ---------------------------------------------------------
DOT_POS = (320, 240)     # Center dot on screen
SAFE_DIST = 200
WARNING_DIST = 120
DANGER_DIST = 60

MIN_HAND_AREA = 6000      # Removes noise
SMOOTHING = 0.4           # Smoother tracking

# For smoothing hand position
last_hand_pos = None


# ---------------------------------------------------------
# Hand Detection using HSV + Motion Mask (FAST)
# ---------------------------------------------------------
def detect_hand(frame, bg_frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range (works for most skin tones)
    lower = np.array([0, 20, 60])
    upper = np.array([20, 255, 255])
    skin_mask = cv2.inRange(hsv, lower, upper)

    # Basic cleanup
    skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)
    skin_mask = cv2.erode(skin_mask, None, iterations=1)
    skin_mask = cv2.dilate(skin_mask, None, iterations=2)

    # Motion detection to improve accuracy
    diff = cv2.absdiff(frame, bg_frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, motion_mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

    # Combine skin + motion
    final_mask = cv2.bitwise_and(skin_mask, motion_mask)

    # Find contours
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, final_mask

    biggest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(biggest) < MIN_HAND_AREA:
        return None, final_mask

    M = cv2.moments(biggest)
    if M["m00"] == 0:
        return None, final_mask

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy), final_mask


# ---------------------------------------------------------
# Safety State Calculation
# ---------------------------------------------------------
def get_state(dist):
    if dist > SAFE_DIST:
        return "SAFE", (0, 255, 0)
    elif dist > WARNING_DIST:
        return "WARNING", (0, 255, 255)
    elif dist > DANGER_DIST:
        return "DANGER", (0, 128, 255)
    else:
        return "DANGER DANGER", (0, 0, 255)


# ---------------------------------------------------------
# Main Application
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)

# Ensure good FPS
cap.set(cv2.CAP_PROP_FPS, 30)

# Initial background frame for motion mask
_, bg_frame = cap.read()
bg_frame = cv2.GaussianBlur(bg_frame, (21, 21), 0)

fps_timer = time.time()
frame_count = 0

print("Running... Press Q to exit")

# Define frame dimensions
WIDTH = 640
HEIGHT = 480

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Smooth background update (adaptive background)
    bg_frame = cv2.addWeighted(frame, 0.05, bg_frame, 0.95, 0)

    # Detect hand
    hand_pos, mask = detect_hand(frame, bg_frame)

    # Draw dot
    cv2.circle(frame, DOT_POS, 10, (255, 0, 0), -1)

    # Hand exists
    if hand_pos:
        cx, cy = hand_pos

        # Smooth the hand movement (reduces jitter)
        if last_hand_pos is None:
            last_hand_pos = np.array([cx, cy], dtype="float32")
        else:
            last_hand_pos = SMOOTHING * last_hand_pos + (1 - SMOOTHING) * np.array([cx, cy])

        cx, cy = int(last_hand_pos[0]), int(last_hand_pos[1])

        # Draw hand position
        cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)
        cv2.line(frame, (cx, cy), DOT_POS, (255, 255, 255), 2)

        # Distance
        dist = int(np.linalg.norm(np.array([cx - DOT_POS[0], cy - DOT_POS[1]])))

        # State label
        state, color = get_state(dist)
        cv2.putText(frame, state, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    # ---- processing code ----
    hand_pos, mask = detect_hand(frame, bg_frame)
    frame_count += 1

    # ---- FPS calculation ----
    if frame_count >= 10:
        elapsed = time.time() - fps_timer
        fps = frame_count / elapsed
        fps_timer = time.time()
        frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, HEIGHT - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # ---- DISPLAY ----
    cv2.imshow("Hand Safety System", frame)
    cv2.imshow("Mask", mask)

    # ---- QUIT ----
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

