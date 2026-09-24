import json
import subprocess
import sys
import os


# Get image path from command line
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"Image not found: {IMAGE_PATH}")
    sys.exit(1)


print("\n================================")
print("NFT MOE PIPELINE")
print("================================")

print(f"\nInput image: {IMAGE_PATH}")


# --------------------------------------------------
# STEP 1 - Run Multi-YOLO
# --------------------------------------------------

print("\n[1/4] Running YOLO experts...")

result = subprocess.run(
    [sys.executable, "multi_yolo.py", IMAGE_PATH],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("YOLO pipeline failed.")
    print(result.stdout)
    print(result.stderr)
    sys.exit(1)

print("YOLO experts completed.")


# --------------------------------------------------
# STEP 2 - Match detections
# --------------------------------------------------

print("\n[2/4] Matching detections...")

result = subprocess.run(
    [sys.executable, "match_detections.py"],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("Detection matching failed.")
    print(result.stdout)
    print(result.stderr)
    sys.exit(1)

print("Detection matching completed.")


# --------------------------------------------------
# STEP 3 - Run MoE gating
# --------------------------------------------------

print("\n[3/4] Running MoE gating...")

result = subprocess.run(
    [sys.executable, "moe_detector.py"],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("MoE detector failed.")
    print(result.stdout)
    print(result.stderr)
    sys.exit(1)

print("MoE gating completed.")


# --------------------------------------------------
# STEP 4 - Prepare clean objects
# --------------------------------------------------

print("\n[4/4] Preparing final object list...")

result = subprocess.run(
    [sys.executable, "prepare_objects.py"],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("Object preparation failed.")
    print(result.stdout)
    print(result.stderr)
    sys.exit(1)


# --------------------------------------------------
# Read final objects
# --------------------------------------------------

with open("final_objects.json", "r") as f:
    objects = json.load(f)


print("\n================================")
print("FINAL DETECTED OBJECTS")
print("================================")

for obj in objects:
    print(
        f"{obj['object']} "
        f"({obj['confidence']:.3f})"
    )


print("\n================================")
print("MOE DETECTION PIPELINE COMPLETE")
print("================================")