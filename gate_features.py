import json


# -----------------------------------------
# Load matched objects
# -----------------------------------------

with open("matched_objects.json", "r") as f:
    groups = json.load(f)


# -----------------------------------------
# Convert one object group into features
# -----------------------------------------

def create_features(group):

    features = {
        "yolo_a_conf": 0.0,
        "yolo_b_conf": 0.0,
        "yolo_c_conf": 0.0,

        "yolo_a_present": 0,
        "yolo_b_present": 0,
        "yolo_c_present": 0,

        "class_name": None
    }

    for detection in group:

        expert = detection["expert"]

        confidence = detection["confidence"]

        if features["class_name"] is None:
            features["class_name"] = detection["class_name"]

        if expert == "YOLO-A":

            features["yolo_a_conf"] = confidence
            features["yolo_a_present"] = 1

        elif expert == "YOLO-B":

            features["yolo_b_conf"] = confidence
            features["yolo_b_present"] = 1

        elif expert == "YOLO-C":

            features["yolo_c_conf"] = confidence
            features["yolo_c_present"] = 1

    return features


# -----------------------------------------
# Create feature dataset
# -----------------------------------------

feature_dataset = []

for group in groups:

    features = create_features(group)

    feature_dataset.append(features)


# -----------------------------------------
# Display
# -----------------------------------------

print("\n==============================")
print("GATING FEATURES")
print("==============================")

for i, features in enumerate(
    feature_dataset,
    start=1
):

    print(f"\nObject Group {i}")

    print(
        f"Class: "
        f"{features['class_name']}"
    )

    print(
        f"YOLO-A: "
        f"confidence={features['yolo_a_conf']:.2f}, "
        f"present={features['yolo_a_present']}"
    )

    print(
        f"YOLO-B: "
        f"confidence={features['yolo_b_conf']:.2f}, "
        f"present={features['yolo_b_present']}"
    )

    print(
        f"YOLO-C: "
        f"confidence={features['yolo_c_conf']:.2f}, "
        f"present={features['yolo_c_present']}"
    )


# -----------------------------------------
# Save
# -----------------------------------------

with open(
    "gate_features.json",
    "w"
) as f:

    json.dump(
        feature_dataset,
        f,
        indent=4
    )


print("\n==============================")
print("Saved: gate_features.json")
print("==============================")