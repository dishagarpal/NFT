from ultralytics import YOLO
import cv2

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    # Run YOLO with confidence threshold
    results = model(frame, conf=0.25)

    # Get first result
    result = results[0]

    # Store detected objects
    detected_objects = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        object_name = model.names[class_id]

        detected_objects.append(object_name)

        print(f"{object_name}: {confidence:.2f}")

    # Remove duplicate objects
    detected_objects = list(set(detected_objects))

    print("Detected objects:", detected_objects)
    print("-" * 40)

    # Draw bounding boxes
    annotated_frame = result.plot()

    # Display camera
    cv2.imshow("AI Scene Awareness - Object Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
