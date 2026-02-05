# Unbias – System Architecture

Unbias is designed as a privacy-preserving, uncertainty-aware
computer vision pipeline for participant counting.

## High-Level Flow

1. Input image is processed in-memory
2. People are detected using a pretrained detector
3. Each detected person is cropped
4. Non-identifying visual features are extracted
5. Gender presentation is estimated probabilistically
6. Low-confidence predictions are abstained
7. Results are aggregated statistically
8. Only aggregate outputs are returned

## Core Design Choices

- No facial recognition
- No identity tracking
- Probabilistic outputs instead of hard labels
- Explicit uncertainty handling
- Aggregation-only results

## Architectural Principle

> If the system cannot be trusted in a context,
> it should refuse to speak rather than guess.

This principle is enforced structurally, not by policy.
