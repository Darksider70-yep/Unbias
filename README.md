Unbias
Ethical, Uncertainty-Aware Participant Counting from Images

Overview
Unbias is a computer vision system that estimates participant counts and gender presentation distribution from images without identifying individuals.

Unlike traditional gender-detection systems, Unbias is designed around three principles:

Privacy by architecture — the system cannot identify people even if misused

Uncertainty awareness — the model is allowed to say “I don’t know”

Transparency — limitations are explicit, not hidden

This project was built for a hackathon setting with an emphasis on responsible AI, statistical honesty, and explainability.

What Unbias Does
Given a single image (e.g. classroom, event, crowd):

Detects people in the image

Estimates gender presentation probabilistically

Explicitly handles ambiguous or low-confidence cases

Outputs only aggregated statistics

Provides confidence scores and visual explanations

Example Output
json
Copy code
{
  "total_participants": 47,
  "female_presenting": 18.6,
  "male_presenting": 21.9,
  "uncertain": 6.5,
  "confidence_score": 0.81
}
Counts are probabilistic sums, not forced labels.

What Unbias Does Not Do
❌ No facial recognition

❌ No identity inference

❌ No image storage

❌ No biometric embeddings

❌ No binary gender enforcement

Unbias does not infer gender identity, biological sex, or personal attributes.
It estimates visual gender presentation only, and abstains when uncertain.

Why This Is Different
Most existing systems:

Force binary classification

Hide uncertainty

Use facial features implicitly

Optimize accuracy at the cost of dignity

Unbias is different because it:

Treats uncertainty as a valid outcome

Aggregates probabilities instead of hard labels

Is privacy-safe by design, not policy

Can refuse to make a prediction

Makes its limitations visible in the UI

This is perception modeling — not surveillance.

System Architecture (High-Level)
Person Detection

Pretrained YOLO-based detector (person class only)

Non-Identifying Feature Extraction

Body silhouette

Pose estimation

Clothing texture

Shape & proportion metrics

Probabilistic Gender Presentation Classifier

Outputs:

Female-presenting

Male-presenting

Uncertain

Uncertainty Handling Layer

Confidence thresholding

Entropy-based abstention

Aggregation Engine

Probability-weighted counting

Confidence scoring

Explainability Layer

Heatmaps

Confidence visualizations

Explicit disclaimers

Folder Structure
bash
Copy code
Unbias/
├── app/                # UI & runtime entry
├── core/               # Detection, features, classifier, aggregation
├── privacy/            # Anonymization & data retention logic
├── explainability/     # Heatmaps & confidence reporting
├── evaluation/         # Metrics & bias checks
├── models/             # Pretrained weights
├── pipelines/          # End-to-end inference flow
├── scripts/            # Demo & sanity checks
└── docs/               # Architecture, ethics, pitch
The structure reflects the philosophy: privacy, uncertainty, and explainability are first-class components.

Tech Stack
Language: Python

Computer Vision: OpenCV

Person Detection: YOLO (pretrained)

Modeling: PyTorch

UI Demo: Streamlit or Gradio

API (optional): FastAPI

All components are chosen for:

Fast prototyping

Transparency

CPU-friendly demos

Judge familiarity

Ethical Design Commitments
No individual-level outputs

No persistent data storage

Clear distinction between presentation and identity

Honest failure modes

User-controlled disabling of gender estimation

Unbias is designed to limit harm by design, not by promise.

Use Cases
Classroom or event participation analysis

Diversity estimation (aggregated only)

Research demos on ethical computer vision

Educational examples of uncertainty-aware AI

Not intended for:

Surveillance

Policing

Individual profiling

Known Limitations
Performance varies with lighting and occlusion

Cultural clothing can increase uncertainty

Gender presentation is context-dependent

The model may abstain frequently — by design

Abstention is treated as integrity, not failure.

One-Line Pitch
“Unbias estimates participation without identifying people, proving that computer vision can be powerful, transparent, and humane.”

License & Use
This project is intended for educational, research, and hackathon demonstration purposes only.
Any real-world deployment would require contextual, legal, and ethical review.

Unbias does not claim to see the truth.
It models perception — and knows when to stay silent.

