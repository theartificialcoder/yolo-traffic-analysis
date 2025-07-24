import cv2
from ultralytics import solutions

# --- USER CONFIGURATION ---

# Path to your video file
VIDEO_PATH = "tohuu.mov"

# Path for the output video file
OUTPUT_VIDEO_PATH = "multi_region_counting_output.avi"

# Process only the first N seconds of the video. Set to 0 to process the full video.
TIME_LIMIT_SECONDS = 80

# YOLO model to use for detection
MODEL_NAME = "yolov8x.pt"

# Define your multiple regions here
REGIONS_CONFIG = [
    {
        "name": "Top Region",
        "points": [[877, 261], [1193, 295], [991, 407], [616, 369]],
        "classes": [],
    },
    {
        "name": "Bottom Line",
        "points": [[1379, 245], [1752, 277], [1773, 471], [1156, 422]],
        "classes": [],
    },
    {
        "name": "Side Polygon",
        "points": [[404, 406], [886, 467], [15, 1076], [10, 596]],
        "classes": [],
    }
]

# --- END OF CONFIGURATION ---


# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)
assert cap.isOpened(), f"Error reading video file: {VIDEO_PATH}"

# Get video properties
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Calculate frame limit based on the time limit
if TIME_LIMIT_SECONDS > 0 and fps > 0:
    frame_limit = int(TIME_LIMIT_SECONDS * fps)
    print(f"Processing for {TIME_LIMIT_SECONDS} seconds, which is {frame_limit} frames.")
else:
    frame_limit = -1  # A value of -1 means no limit
    print("No time limit set, processing full video.")

# Initialize object counters
counters = []
for config in REGIONS_CONFIG:
    counter = solutions.ObjectCounter(
        show=False,
        region=config["points"],
        model=MODEL_NAME,
        classes=config["classes"] if config["classes"] else None,
        tracker="bytetrack.yaml",
    )
    counters.append(counter)

# Initialize a frame counter
current_frame_count = 0

# Process the video frame by frame
while cap.isOpened():
    # Stop processing if the frame limit is reached
    if frame_limit != -1 and current_frame_count >= frame_limit:
        print(f"Time limit of {TIME_LIMIT_SECONDS} seconds reached. Stopping processing.")
        break
        
    success, frame = cap.read()
    if not success:
        print("Video frame is empty or processing is complete.")
        break

    # Increment the frame counter
    current_frame_count += 1
    
    # Make a copy of the frame to draw on
    annotated_frame = frame.copy()

    # Loop through each counter and apply it to the frame
    for counter in counters:
        # NEW CORRECT CODE
        counter(annotated_frame)

    # Display the final frame
    cv2.imshow("YOLOv8 Multi-Region Counting", annotated_frame)
    
    # Write the processed frame
    video_writer.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output video saved to: {OUTPUT_VIDEO_PATH}")