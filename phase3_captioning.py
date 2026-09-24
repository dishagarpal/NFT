from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# -----------------------------
# Load BLIP
# -----------------------------
print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

print("BLIP model loaded.\n")


# -----------------------------
# Caption function
# -----------------------------
def generate_caption(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    output = model.generate(
        **inputs,
        max_new_tokens=30
    )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption


# -----------------------------
# Test images
# -----------------------------
images = [
    "test.jpg",
    "multipleperson.jpg",
    "singleperson.jpg"
]


for image_path in images:

    if not os.path.exists(image_path):
        print(f"Skipping {image_path} - file not found.")
        continue

    print(f"Image: {image_path}")

    caption = generate_caption(image_path)

    print(f"Caption: {caption}")
    print("-" * 50)