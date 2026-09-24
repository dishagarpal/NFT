from ultralytics import YOLO
from detection_utils import extract_detections
import json
import sys

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("Loading YOLO-A...")
model_a = YOLO("yolo11n.pt")

print("Loading YOLO-B...")
model_b = YOLO("yolo11s.pt")

print("Loading YOLO-C...")
model_c = YOLO("yolo11m.pt")


# --------------------------------------------------
# Run one expert
# --------------------------------------------------

def run_expert(name, model):

    print(f"\n========== {name} ==========")

    results = model(
        IMAGE_PATH,
        verbose=False
    )

    all_detections = []

    for result in results:

        detections = extract_detections(
            result,
            name
        )

        all_detections.extend(detections)

        for detection in detections:

            print(
                f"{detection['class_name']}: "
                f"{detection['confidence']:.2f} "
                f"box={detection['box']}"
            )

    return all_detections


# --------------------------------------------------
# Run all experts
# --------------------------------------------------

detections_a = run_expert(
    "YOLO-A",
    model_a
)

detections_b = run_expert(
    "YOLO-B",
    model_b
)

detections_c = run_expert(
    "YOLO-C",
    model_c
)


# --------------------------------------------------
# Combine all detections
# --------------------------------------------------

all_detections = (
    detections_a
    + detections_b
    + detections_c
)


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

with open(
    "predictions.json",
    "w"
) as f:

    json.dump(
        all_detections,
        f,
        indent=4
    )


print("\n================================")
print("All detections saved to:")
print("predictions.json")
print("================================")

print(
    f"Total detections: "
    f"{len(all_detections)}"
)