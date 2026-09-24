from ultralytics import YOLO

print("Preparing COCO128...")

# Loading the model causes Ultralytics to prepare/download
# the dataset when it is first used.
model = YOLO("yolo11n.pt")

model.val(
    data="coco128.yaml",
    split="val"
)

print("\nCOCO128 is ready.")