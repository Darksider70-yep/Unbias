# scripts/run_demo.py

import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from pipelines.inference_pipeline import InferencePipeline

pipeline = InferencePipeline()

img_path = os.path.join(os.path.dirname(__file__), "sample.jpg")
image = cv2.imread(img_path)

if image is None:
    print("Error: Image could not be loaded.")
    sys.exit(1)

result = pipeline.run(image)

print("\nUnbias Output:")
print(result)
