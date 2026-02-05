# Ethical Considerations – Unbias

Unbias is built with the assumption that
visual inference about humans is inherently uncertain and context-dependent.

## What We Explicitly Avoid

- Inferring gender identity
- Inferring biological sex
- Identifying individuals
- Persisting personal data
- Binary classification enforcement

## Gender Presentation vs Identity

Unbias estimates **gender presentation** —
how a person may visually appear in a given context.

This is not a claim about who a person is.

## Privacy by Design

- Images are processed in-memory only
- No facial embeddings are created
- No raw images are stored
- Outputs are aggregated statistics only

## Failure as Integrity

Abstaining from prediction is treated as
a sign of model integrity, not weakness.

Uncertainty is surfaced, not hidden.
