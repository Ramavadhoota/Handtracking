Hand Safety — Real-time Hand Proximity Prototype

Prototype: Track the user’s hand from a live camera feed and detect when it approaches a virtual object on the screen.
Goal: show an on-screen warning "DANGER DANGER" when the hand reaches a danger boundary.
Constraints: No MediaPipe / OpenPose / cloud APIs. Allowed: OpenCV, NumPy, (optional) PyTorch/TensorFlow for small models.
Target: real-time CPU execution (≥ 8 FPS).

Table of Contents

Features

How it works (overview)

Requirements

Installation (quick)

Run (live camera)

Configuration / Tuning

Expected Output & UI

Performance Tips (reach ≥ 8 FPS)

Troubleshooting

Extensions & Next Steps

Project structure suggestion

License & Credits

Features

Real-time hand (or fingertip) detection using classical CV techniques (no external pose APIs).

Virtual object / boundary rendered on screen (configurable).

Distance-based dynamic state logic: SAFE / WARNING / DANGER (and clear "DANGER DANGER" overlay in danger).

Smoothing & motion fusion for stable centroid tracking.

Designed to meet CPU-only real-time performance (target ≥ 8 FPS).

Lightweight and easy to run locally (VS Code / terminal).

How it works (overview)

Capture frames from a live webcam (OpenCV).

Preprocess (resize, blur).

Detect hand via a combination of:

Color segmentation (HSV skin / glove color range), and

Motion mask / background subtraction to reduce false positives (optional but recommended).

Find contours and select the largest plausible hand contour. Compute the centroid (or fingertip if you detect extreme points).

Compute distance between the hand centroid and a virtual object point/shape.

Map distance → state:

dist > SAFE_DIST → SAFE

WARNING_DIST < dist <= SAFE_DIST → WARNING

dist <= WARNING_DIST → DANGER (show "DANGER DANGER")

Overlay UI (dot, line to hand, state text, FPS).

Optional: log events, sound alarm, record video.

This is intentionally simple and fast (no heavy deep models), robust for controlled indoor lighting; optionally swap in a tiny PyTorch hand detector if needed.

Requirements

Minimum:

Python 3.8+

OpenCV (opencv-python)

NumPy

Install with pip:

pip install opencv-python numpy


Optional (audio alarms, logging):

pip install playsound


If you later add a PyTorch model:

pip install torch torchvision

Installation (quick)

Clone or copy the project into a folder.

(Recommended) Create a virtual environment:

python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows


Install deps:

pip install opencv-python numpy


Open in VS Code or your preferred editor.

Run (live camera)

Save the provided script as hand_safety_realtime.py (or use the sample file in this repo). Then:

python hand_safety_realtime.py


A window titled Hand Safety System will open, and Mask (detection mask) will show optionally.

Press Q in the window to quit.

Configuration & Tuning

The script exposes several parameters at the top — tune for your camera, environment, and desired sensitivity:

DOT_POS = (320, 240)     # (x, y) screen position of the virtual object
SAFE_DIST = 200          # px distance threshold for SAFE
WARNING_DIST = 120       # px threshold for WARNING
DANGER_DIST = 60         # px threshold for DANGER DANGER

MIN_HAND_AREA = 6000     # min contour area to accept as hand
SMOOTHING = 0.4          # 0.0 .. 1.0; smoothing factor for centroid
FRAME_WIDTH = 640        # lower this to increase FPS (320 recommended for slow CPUs)
FRAME_HEIGHT = 480

Skin color / glove color (HSV)

If skin detection is unreliable, use a colored glove and change HSV bounds:

# Example for red glove
lower = np.array([0, 120, 70])
upper = np.array([10, 255, 255])
# or for orange / bright neon adjust accordingly


If you have diverse skin tones, widen ranges (but that increases false positives). Use motion fusion (diff vs background) to reduce false alarms.

Expected Output & UI

Live camera overlay includes:

Virtual object (dot) on the video.

Hand centroid (circle) and line to the dot.

Text state: SAFE / WARNING / DANGER (colored).

When in danger, large flashing "DANGER DANGER" overlay.

FPS readout.

Optional debug window for the detection mask.

Performance Tips (reach ≥ 8 FPS)

Lower frame resolution — use 320x240 for slow CPUs.

Use lightweight masks: small kernels for morphological ops; avoid heavy per-frame computations.

Motion fusion: combining skin mask with a cheap motion mask reduces false work and keeps tracking stable.

Smoothing: reduces jitter but is cheap.

Avoid per-frame visual I/O costs: do not write frames to disk during runtime.

Use optimized OpenCV build when available (e.g., pip wheels with optimizations).

Measure: use the FPS counter in the script to monitor performance and tune parameters.

Typical settings (modern laptop CPU):

FRAME_WIDTH = 640 → ~15–30 FPS

FRAME_WIDTH = 320 → easily ≥ 30 FPS (very safe for ≥ 8 FPS)

Troubleshooting

Camera does not open:

Check camera being used by other apps.

Ensure cv2.VideoCapture(0) index is correct. Try 1, 2, … if multiple devices.

Run quick test:

python - <<EOF
import cv2
print(cv2.__version__)
cap = cv2.VideoCapture(0)
print("Opened:", cap.isOpened())
cap.release()
EOF


No hand detected / many false positives:

Improve lighting and background contrast.

Use a colored glove for robust detection.

Tune HSV ranges and MIN_HAND_AREA.

Enable motion mask (the script uses background diff).

Low FPS:

Reduce FRAME_WIDTH / FRAME_HEIGHT.

Remove or reduce debug windows (cv2.imshow("Mask", mask)).

Close other CPU-heavy apps.

break outside loop` or indentation errors:

Make sure the display and waitKey block is inside your main while True: loop with correct indentation.

Extensions & Next Steps

Ideas to extend or improve the POC:

Replace classical detector with a small, faster learned detector (tiny YOLOv5/YOLOv8s trained on hand dataset) for robust cluttered scenes — still possible to run on CPU if small and pruned.

Add fingertip detection by analyzing extreme contour points or convex hull defects.

Multi-hand tracking and choose the hand nearest the target.

Log events (timestamped) to CSV for analysis.

Sound alert or visual flash (blink) when danger is triggered.

Add bounding box tracking and Kalman filter for smoother multi-frame tracking.

Project structure suggestion
hand-safety/
├─ hand_safety_realtime.py      # main script (live camera)
├─ process_video.py             # offline video processing variant
├─ requirements.txt
├─ README.md
└─ assets/
   └─ sample_videos/            # optional test videos


requirements.txt example:

opencv-python
numpy
playsound   # optional

Example run command
python hand_safety_realtime.py


If you want to process a recorded video instead (Colab or offline):

python process_video.py --input sample_videos/hand_test.mp4 --output out.mp4


(You can include an argument parser to accept --width, --dot_x, etc.)
