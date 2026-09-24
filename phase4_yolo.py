from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

print("YOLO model loaded.")

# Run YOLO on the same image used for Phase 3
results = model("test.jpg", conf=0.25)

for result in results:

    print("\nDetected Objects:")

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.2f}")

    # Show bounding boxes
    result.show()