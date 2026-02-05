# Unbias Privacy Policy (Architectural)

Unbias enforces the following constraints by design:

## The System Will Not

- Store raw images
- Store cropped person images
- Perform face recognition
- Generate identity embeddings
- Output individual-level predictions

## Data Handling

- All image data is processed in-memory
- Temporary data is deleted immediately after inference
- Only aggregate numerical outputs are returned

## Non-Negotiable Principle

Even with modification,
the core architecture prevents individual identification.

Privacy is not configurable.
