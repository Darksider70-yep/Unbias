# scripts/run_demo.py

import cv2
import sys
from pipelines.inference_pipeline import InferencePipeline

pipeline = InferencePipeline()

import os

img_path = os.path.join(os.path.dirname(__file__), "sample.jpg")
image = cv2.imread(img_path)

if image is None:
    print("Error: Image could not be loaded.")
    sys.exit(1)

result = pipeline.run(image)

print("\nUnbias Output:")
print(result)
