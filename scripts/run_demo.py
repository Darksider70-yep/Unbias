# scripts/run_demo.py

import cv2
import sys
from pipelines.inference_pipeline import InferencePipeline

pipeline = InferencePipeline()

image = cv2.imread("scripts/sample.jpg")

if image is None:
    print("Error: Image could not be loaded.")
    sys.exit(1)

result = pipeline.run(image)

print("\nUnbias Output:")
print(result)
