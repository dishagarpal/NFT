import json
import os
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "datasets/coco128"
IMAGE_DIR = os.path.join(DATASET_PATH, "images", "train2017")
LABEL_DIR = os.path.join(DATASET_PATH, "labels", "train2017")

OUTPUT_FILE = "gate_training.json"

IOU_THRESHOLD = 0.5

MODELS = {
    "A": "yolo11n.pt",
    "B": "yolo11s.pt",
    "C": "yolo11m.pt"
}


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading YOLO models...")

models = {}

for name, path in MODELS.items():

    print(f"Loading YOLO-{name}...")

    models[name] = YOLO(path)


print("All models loaded.")


# ============================================================
# IOU FUNCTION
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# LOAD GROUND-TRUTH LABEL
# ============================================================

def load_labels(label_file):

    objects = []

    if not os.path.exists(label_file):
        return objects

    with open(label_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])

            width = float(parts[3])
            height = float(parts[4])

            objects.append({
                "class_id": class_id,
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })

    return objects


# ============================================================
# CONVERT YOLO BOX TO PIXEL BOX
# ============================================================

def yolo_to_pixel_box(obj, image_width, image_height):

    xc = obj["x_center"] * image_width
    yc = obj["y_center"] * image_height

    w = obj["width"] * image_width
    h = obj["height"] * image_height

    x1 = xc - w / 2
    y1 = yc - h / 2

    x2 = xc + w / 2
    y2 = yc + h / 2

    return [
        x1,
        y1,
        x2,
        y2
    ]


# ============================================================
# GET DETECTIONS FOR ONE IMAGE
# ============================================================

def get_detections(model, image_path):

    results = model(
        image_path,
        verbose=False
    )

    detections = []

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for box, class_id, confidence in zip(
            boxes,
            classes,
            confidences
        ):

            detections.append({
                "class_id": int(class_id),
                "confidence": float(confidence),
                "box": box
            })

    return detections


# ============================================================
# BUILD CLASS-SPECIFIC FEATURES
# ============================================================

def get_class_features(
    detections,
    gt_box,
    gt_class_id
):

    best_iou = 0.0
    best_confidence = 0.0

    for detection in detections:

        # IMPORTANT:
        # Only consider predictions for the
        # candidate/ground-truth class.
        if detection["class_id"] != gt_class_id:
            continue

        iou = calculate_iou(
            detection["box"],
            gt_box
        )

        if iou >= IOU_THRESHOLD:

            if (
                iou > best_iou
                or
                (
                    iou == best_iou
                    and
                    detection["confidence"] >
                    best_confidence
                )
            ):

                best_iou = iou

                best_confidence = (
                    detection["confidence"]
                )

    present = 1 if best_iou >= IOU_THRESHOLD else 0

    return {
        "confidence": best_confidence,
        "present": present,
        "iou": best_iou
    }


# ============================================================
# PROCESS DATASET
# ============================================================

print("\n================================")
print("CREATING GATE DATASET")
print("================================")

image_files = [

    f for f in os.listdir(IMAGE_DIR)

    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )

]

print(
    "Images found:",
    len(image_files)
)


training_data = []


for image_index, image_name in enumerate(
    image_files,
    start=1
):

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    label_name = os.path.splitext(
        image_name
    )[0] + ".txt"

    label_path = os.path.join(
        LABEL_DIR,
        label_name
    )

    ground_truth_objects = load_labels(
        label_path
    )

    if not ground_truth_objects:
        continue


    # --------------------------------------------------------
    # Run all three YOLO experts
    # --------------------------------------------------------

    detections = {}

    for expert_name, model in models.items():

        detections[expert_name] = (
            get_detections(
                model,
                image_path
            )
        )


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    from PIL import Image

    with Image.open(image_path) as image:

        image_width, image_height = image.size


    # --------------------------------------------------------
    # Process every ground-truth object
    # --------------------------------------------------------

    for gt in ground_truth_objects:

        gt_class_id = gt["class_id"]

        gt_box = yolo_to_pixel_box(
            gt,
            image_width,
            image_height
        )


        features = {}


        expert_ious = {}


        # ----------------------------------------------------
        # YOLO-A
        # ----------------------------------------------------

        a = get_class_features(
            detections["A"],
            gt_box,
            gt_class_id
        )

        features["a_conf"] = a["confidence"]
        features["a_present"] = a["present"]

        expert_ious["A"] = a["iou"]


        # ----------------------------------------------------
        # YOLO-B
        # ----------------------------------------------------

        b = get_class_features(
            detections["B"],
            gt_box,
            gt_class_id
        )

        features["b_conf"] = b["confidence"]
        features["b_present"] = b["present"]

        expert_ious["B"] = b["iou"]


        # ----------------------------------------------------
        # YOLO-C
        # ----------------------------------------------------

        c = get_class_features(
            detections["C"],
            gt_box,
            gt_class_id
        )

        features["c_conf"] = c["confidence"]
        features["c_present"] = c["present"]

        expert_ious["C"] = c["iou"]


        # ----------------------------------------------------
        # SELECT TARGET EXPERT
        # ----------------------------------------------------

        best_expert = max(
            expert_ious,
            key=expert_ious.get
        )

        best_iou = expert_ious[
            best_expert
        ]


        # If no expert detects the object,
        # skip this example.

        if best_iou < IOU_THRESHOLD:

            continue


        # ----------------------------------------------------
        # SAVE TRAINING EXAMPLE
        # ----------------------------------------------------

        training_data.append({

            "image": image_name,

            "ground_truth_class": gt_class_id,

            "features": features,

            "target": best_expert

        })


    if image_index % 25 == 0:

        print(
            f"Processed "
            f"{image_index}/"
            f"{len(image_files)} images | "
            f"Examples: "
            f"{len(training_data)}"
        )


# ============================================================
# SAVE DATASET
# ============================================================

with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        training_data,
        f,
        indent=4
    )


# ============================================================
# STATISTICS
# ============================================================

target_counts = {
    "A": 0,
    "B": 0,
    "C": 0
}


for example in training_data:

    target_counts[
        example["target"]
    ] += 1


print("\n================================")
print("GATE DATASET CREATED")
print("================================")

print(
    "Total examples:",
    len(training_data)
)

print(
    "YOLO-A targets:",
    target_counts["A"]
)

print(
    "YOLO-B targets:",
    target_counts["B"]
)

print(
    "YOLO-C targets:",
    target_counts["C"]
)

print(
    "Saved:",
    OUTPUT_FILE
)