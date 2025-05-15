import cv2  # OpenCV library for computer vision operations
from ultralytics import YOLO  # YOLO object detection model
from pathlib import Path  # For cross-platform path handling

# Define the path to the trained model weights
# Using Path ensures cross-platform compatibility (Windows/Linux/Mac)
model_path = Path("uno_train3") / "train" / "weights" / "best.pt"
model = YOLO(str(model_path))  # Load the YOLO model with trained weights

# Dictionary mapping internal model class names to human-readable display names
name_mapping = {
    # Light Blue cards
    'lblue-1': 'light blue 1', 'lblue-2': 'light blue 2', 'lblue-3': 'light blue 3',
    'lblue-4': 'light blue 4', 'lblue-5': 'light blue 5', 'lblue-6': 'light blue 6',
    'lblue-7': 'light blue 7', 'lblue-8': 'light blue 8', 'lblue-9': 'light blue 9',
    'lblue-10': 'light blue 0',  # Assuming '10' represents 0 in some datasets
    'lblueflip-20': 'light blue Flip', 'lbluerevers-20': 'light blue Reverse',
    'lblueskip-20': 'light blue Skip',

    # Light Green cards
    'lgreen-1': 'light green 1', 'lgreen-2': 'light green 2', 'lgreen-3': 'light green 3',
    'lgreen-4': 'light green 4', 'lgreen-5': 'light green 5', 'lgreen-6': 'light green 6',
    'lgreen-7': 'light green 7', 'lgreen-8': 'light green 8', 'lgreen-9': 'light green 9',
    'lgreen-10': 'light green 0',
    'lgreenflip-20': 'light green Flip', 'lgreenrevers-20': 'light green Reverse',
    'lgreenskip-20': 'light green Skip',

    # Light Red cards
    'lred-1': 'light red 1', 'lred-2': 'light red 2', 'lred-3': 'light red 3',
    'lred-4': 'light red 4', 'lred-5': 'light red 5', 'lred-6': 'light red 6',
    'lred-7': 'light red 7', 'lred-8': 'light red 8', 'lred-9': 'light red 9',
    'lred-10': 'light red 0',
    'lredflip-20': 'light red Flip', 'lredrevers-20': 'light red Reverse',
    'lredskip-20': 'light red Skip',

    # Light Yellow cards
    'lyellow-1': 'light yellow 1', 'lyellow-2': 'light yellow 2', 'lyellow-3': 'light yellow 3',
    'lyellow-4': 'light yellow 4', 'lyellow-5': 'light yellow 5', 'lyellow-6': 'light yellow 6',
    'lyellow-7': 'light yellow 7', 'lyellow-8': 'light yellow 8', 'lyellow-9': 'light yellow 9',
    'lyellow-10': 'light yellow 0',
    'lyellowflip-20': 'light yellow Flip', 'lyellowrevers-20': 'light yellow Reverse',
    'lyellowskip-20': 'light yellow Skip',

    # Special
    'lcolor-40': 'Wild',
    'lcolor2-50': 'Wild Draw Four',
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
    cv2.imshow("UNO Card Detector", frame)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
