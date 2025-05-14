import cv2  # OpenCV library for computer vision operations
from ultralytics import YOLO  # YOLO object detection model
from pathlib import Path  # For cross-platform path handling

# Define the path to the trained model weights
# Using Path ensures cross-platform compatibility (Windows/Linux/Mac)
model_path = Path("uno_train1") / "train22" / "weights" / "best.pt"
model = YOLO(str(model_path))  # Load the YOLO model with trained weights

# Dictionary mapping internal model class names to human-readable display names
name_mapping = {
    # Blue cards
    "blue-0": "blue 0", "blue-1": "blue 1", "blue-2": "blue 2", "blue-20": "blue Draw two", "blue-3": "blue 3",
    "blue-4": "blue 4", "blue-5": "blue 5", "blue-6": "blue 6", "blue-7": "blue 7", "blue-8": "blue 8", 
    "blue-9": "blue 9", "bluereverse-20": "blue reverse", "blueskip-20": "blue skip",
    # Special cards
    "color-40": "Wild", "color-400": "Wild draw 4",
    # Green cards
    "green-0": "green 0", "green-1": "green 1", "green-2": "green 2", "green-20": "green Draw two", "green-3": "green 3",
    "green-4": "green 4", "green-5": "green 5", "green-6": "green 6", "green-7": "green 7", "green-8": "green 8", 
    "green-9": "green 9", "greenreverse-20": "green reverse", "greenskip-20": "green skip",
    # Red cards
    "red-0": "red 0", "red-1": "red 1", "red-2": "red 2", "red-20": "red Draw two", "red-3": "red 3", "red-4": "red 4",
    "red-5": "red 5", "red-6": "red 6", "red-7": "red 7", "red-8": "red 8", "red-9": "red 9", "redreverse-20": "red reverse",
    "redskip-20": "red skip", 
    # Yellow cards
    "yellow-0": "yellow 0", "yellow-1": "yellow 1", "yellow-2": "yellow 2", "yellow-20": "yellow Draw two", "yellow-3": "yellow 3",
    "yellow-4": "yellow 4", "yellow-5": "yellow 5", "yellow-6": "yellow 6", "yellow-7": "yellow 7", "yellow-8": "yellow 8", 
    "yellow-9": "yellow 9", "yellowreverse-20": "yellow reverse", "yellowskip-20": "yellow skip",
}

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Main loop for continuous video capture and processing
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection on the current frame
    results = model(frame)

    # Process each detection result
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates (x1,y1,x2,y2)
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores for each detection
        class_ids = result.boxes.cls.cpu().numpy()  # Class IDs for each detected object

        # Process each detected object
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            # Extract coordinates for bounding box
            x1, y1, x2, y2 = box
            # Get the original label from model's class names
            original_label = model.names[int(class_id)]

            # Convert technical label to human-readable format
            label = name_mapping.get(original_label, original_label)
            # Add confidence score to label
            label = f"{label}: {conf:.2f}"

            # Draw detection visualization on frame
            # Green rectangle around detected object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Label text above rectangle
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the processed frame with detections
    cv2.imshow("YOLO Inference", frame)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
