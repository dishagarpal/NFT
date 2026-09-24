from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
import os
import sys


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

OBJECT_FILE = "final_objects.json"
OUTPUT_FILE = "final_description.txt"


# ============================================================
# CHECK IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):

    print(f"Image not found: {IMAGE_PATH}")
    sys.exit(1)


# ============================================================
# LOAD BLIP
# ============================================================

print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

print("BLIP model loaded.")


# ============================================================
# LOAD MOE OBJECTS
# ============================================================

if not os.path.exists(OBJECT_FILE):

    print("final_objects.json not found.")
    print("Run moe_pipeline.py first.")
    sys.exit(1)


with open(
    OBJECT_FILE,
    "r"
) as f:

    objects = json.load(f)


print("\n================================")
print("MOE DETECTED OBJECTS")
print("================================")


for obj in objects:

    print(
        f"- {obj['object']}: "
        f"{obj['confidence']:.2f}"
    )


# ============================================================
# REMOVE DUPLICATE OBJECT TYPES
# ============================================================

unique_objects = []

seen = set()


for obj in objects:

    object_name = obj["object"]

    if object_name not in seen:

        unique_objects.append(
            object_name
        )

        seen.add(object_name)


# ============================================================
# RUN BLIP
# ============================================================

print("\nRunning BLIP...")

image = Image.open(
    IMAGE_PATH
).convert("RGB")


inputs = processor(
    images=image,
    return_tensors="pt"
)


output = model.generate(
    **inputs,
    max_new_tokens=30
)


blip_caption = processor.decode(
    output[0],
    skip_special_tokens=True
)


print("\n================================")
print("BLIP CAPTION")
print("================================")

print(blip_caption)


# ============================================================
# CREATE MOE-GROUNDED DESCRIPTION
# ============================================================

print("\n================================")
print("FINAL NATURAL DESCRIPTION")
print("================================")


# Separate object categories

people = [
    obj for obj in unique_objects
    if obj == "person"
]

other_objects = [
    obj for obj in unique_objects
    if obj != "person"
]


# Count people

person_count = sum(
    1 for obj in objects
    if obj["object"] == "person"
)


# ------------------------------------------------------------
# Build natural description
# ------------------------------------------------------------

sentences = []


if person_count == 1:

    sentences.append(
        "There is one person"
    )

elif person_count > 1:

    sentences.append(
        f"There are {person_count} people"
    )


# ------------------------------------------------------------
# Describe detected objects
# ------------------------------------------------------------

if other_objects:

    readable_objects = []

    for obj in other_objects:

        if obj == "dining table":

            readable_objects.append(
                "a dining table"
            )

        elif obj == "tv":

            readable_objects.append(
                "a TV"
            )

        elif obj == "bottle":

            readable_objects.append(
                "a bottle"
            )

        elif obj == "cup":

            readable_objects.append(
                "a cup"
            )

        elif obj == "clock":

            readable_objects.append(
                "a clock"
            )

        else:

            readable_objects.append(
                f"a {obj}"
            )


    if len(readable_objects) == 1:

        object_sentence = (
            f" and {readable_objects[0]}"
        )

    elif len(readable_objects) == 2:

        object_sentence = (
            f" and {readable_objects[0]} "
            f"and {readable_objects[1]}"
        )

    else:

        object_sentence = (
            " and "
            + ", ".join(
                readable_objects[:-1]
            )
            + ", and "
            + readable_objects[-1]
        )


    if sentences:

        sentences[0] += object_sentence + " are visible."

    else:

        sentences.append(
            "Visible objects include "
            + ", ".join(readable_objects)
            + "."
        )


else:

    if sentences:

        sentences[0] += "."

    else:

        sentences.append(
            "No recognized objects were detected."
        )


final_description = " ".join(
    sentences
)


print(final_description)


# ============================================================
# SAVE DESCRIPTION
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        final_description
    )


print("================================")
print(
    f"Saved: {OUTPUT_FILE}"
)