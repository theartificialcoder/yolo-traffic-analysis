import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# --- Configuration ---
VIDEO_PATH = "/home/theartificialcoder/phd/Traffic Dataset/MCT - Nguyen Co Thach/AM Peak (1)/269 - MCTho - NCThach 2 - 2025-01-10 07-30-14-939.mov"
OUTPUT_VIDEO_PATH = "tracked_video_multi_zone.mp4"
MODEL_NAME = 'yolov8n.pt'

# --- DEFINE YOUR MULTIPLE ZONES ---
# Paste the coordinates you got from Roboflow for each zone.
# You can add as many zones as you like to this list.
ZONES_CONFIG = [
    {
        "name": "Zone A",
        "polygon": np.array([[877, 261], [1193, 295], [991, 407], [616, 369]]),
        "color": sv.Color.red()
    },
    {
        "name": "Zone B",
        "polygon":  np.array([[1379, 245], [1752, 277], [1773, 471], [1156, 422]]),
        "color": sv.Color.blue()
    },
    {
        "name": "Zone C",
        "polygon":  np.array([[488, 423], [886, 467], [395, 796], [8, 686]]),
        "color": sv.Color.yellow()
    }
]

# --- Main Processing ---

# Load the YOLOv8 model
model = YOLO(MODEL_NAME)

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Get video properties for the VideoWriter and Zone setup
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_resolution_wh = (frame_width, frame_height)

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, video_resolution_wh)

# Initialize the zones and annotators from the configuration
zones = []
zone_annotators = []
for config in ZONES_CONFIG:
    zone = sv.PolygonZone(polygon=config["polygon"], frame_resolution_wh=video_resolution_wh)
    zones.append(zone)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=config["color"],
        thickness=2,
        text_thickness=2,
        text_scale=1
    )
    zone_annotators.append(zone_annotator)

# Create a default box annotator for all detections
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=1,
    text_scale=0.5
)

print(f"Processing video... Output will be saved to {OUTPUT_VIDEO_PATH}")

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 tracking on the frame
    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml")
    detections = sv.Detections.from_ultralytics(results[0])

    # Create labels for all detections
    labels = [
        f"#{tracker_id} {model.model.names[class_id]}"
        for _, _, confidence, class_id, tracker_id
        in detections
    ]
    
    # Annotate the frame with default bounding boxes for all detections
    annotated_frame = box_annotator.annotate(
        scene=frame.copy(), 
        detections=detections,
        labels=labels
    )
    
    # Loop through each zone to check detections and annotate
    for i, zone in enumerate(zones):
        # Get a boolean mask of detections inside the current zone
        mask = zone.trigger(detections=detections)
        
        # Annotate the polygon zone itself (with count)
        annotated_frame = zone_annotators[i].annotate(scene=annotated_frame)
    
    # Write the frame to the output video
    video_writer.write(annotated_frame)
    
    # Display the frame
    cv2.imshow("YOLOv8 Multi-Zone Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Finalization ---
cap.release()
video_writer.release()
cv2.destroyAllWindows()

print(f"Processing complete. Tracked video saved to {OUTPUT_VIDEO_PATH}")