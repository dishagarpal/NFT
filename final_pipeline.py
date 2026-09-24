import subprocess
import sys
import os


# --------------------------------------------------
# Get image path
# --------------------------------------------------

IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"

if not os.path.exists(IMAGE_PATH):
    print(f"Image not found: {IMAGE_PATH}")
    sys.exit(1)


print("\n============================================================")
print("NFT COMPLETE PIPELINE")
print("============================================================")

print(f"\nInput image: {IMAGE_PATH}")


# --------------------------------------------------
# STEP 1 - MOE OBJECT DETECTION
# --------------------------------------------------

print("\n============================================================")
print("STEP 1/3 - MOE OBJECT DETECTION")
print("============================================================")

result = subprocess.run(
    [sys.executable, "moe_pipeline.py", IMAGE_PATH],
    check=True
)


# --------------------------------------------------
# STEP 2 - BLIP DESCRIPTION
# --------------------------------------------------

print("\n============================================================")
print("STEP 2/3 - BLIP DESCRIPTION")
print("============================================================")

result = subprocess.run(
    [sys.executable, "phase4_combined.py", IMAGE_PATH],
    check=True
)


# --------------------------------------------------
# STEP 3 - TEXT TO SPEECH
# --------------------------------------------------

print("\n============================================================")
print("STEP 3/3 - TEXT TO SPEECH")
print("============================================================")

result = subprocess.run(
    [sys.executable, "test_tts.py"],
    check=True
)


print("\n============================================================")
print("FINAL NFT PIPELINE COMPLETE")
print("============================================================")