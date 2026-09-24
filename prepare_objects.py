import json


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "final_detections.json"
OUTPUT_FILE = "final_objects.json"

# General minimum confidence
MIN_CONFIDENCE = 0.18

# Person detections need stronger evidence
PERSON_MIN_CONFIDENCE = 0.40


# ============================================================
# LOAD MOE RESULTS
# ============================================================

with open(INPUT_FILE, "r") as f:
    detections = json.load(f)


objects = []


# ============================================================
# FILTER DETECTIONS
# ============================================================

for detection in detections:

    object_name = detection["object"]
    confidence = float(detection["confidence"])


    # --------------------------------------------------------
    # Person requires higher confidence
    # --------------------------------------------------------

    if object_name == "person":

        if confidence < PERSON_MIN_CONFIDENCE:
            continue


    # --------------------------------------------------------
    # Other objects
    # --------------------------------------------------------

    else:

        if confidence < MIN_CONFIDENCE:
            continue


    objects.append({

        "object": object_name,

        "confidence": round(
            confidence,
            3
        )

    })


# ============================================================
# SAVE CLEAN OUTPUT
# ============================================================

with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        objects,
        f,
        indent=4
    )


# ============================================================
# PRINT RESULT
# ============================================================

print("\n==============================")
print("FILTERED FINAL OBJECT LIST")
print("==============================")


if not objects:

    print("No reliable objects detected.")


else:

    for obj in objects:

        print(
            f"{obj['object']} "
            f"({obj['confidence']:.3f})"
        )


print(
    "\nSaved:",
    OUTPUT_FILE
)