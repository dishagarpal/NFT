import json

from iou_utils import calculate_iou


IOU_THRESHOLD = 0.5


# ---------------------------------------------
# Load predictions
# ---------------------------------------------

with open("predictions.json", "r") as f:
    detections = json.load(f)


# ---------------------------------------------
# Group detections based ONLY on box overlap
# Do NOT require the class names to match.
# ---------------------------------------------

groups = []


for detection in detections:

    matched_group = None

    for group in groups:

        for existing in group:

            iou = calculate_iou(
                detection["box"],
                existing["box"]
            )

            if iou >= IOU_THRESHOLD:

                matched_group = group
                break

        if matched_group is not None:
            break


    if matched_group is not None:

        matched_group.append(detection)

    else:

        groups.append([detection])


# ---------------------------------------------
# Print results
# ---------------------------------------------

print("\n==============================")
print("MATCHED OBJECT GROUPS")
print("==============================")


for i, group in enumerate(groups, start=1):

    print(f"\nObject Group {i}")

    for detection in group:

        print(
            f"  {detection['expert']} -> "
            f"{detection['class_name']} -> "
            f"{detection['confidence']:.2f}"
        )


# ---------------------------------------------
# Save
# ---------------------------------------------

with open(
    "matched_objects.json",
    "w"
) as f:

    json.dump(
        groups,
        f,
        indent=4
    )


print("\n==============================")
print("Saved: matched_objects.json")
print("==============================")