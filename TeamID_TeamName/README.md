# Gender Classification Submission

This folder is structured for the event evaluator.

## Label Mapping
- `0` -> Male
- `1` -> Female

## Runtime Constraints
- PyTorch model format: `model/model.pth`
- CPU-only inference
- Offline execution (no internet usage)
- Python 3.10 compatible

## Required API
`inference.py` exposes:

```python
def predict(image_path):
    """
    Returns:
    label: int (0 = Male, 1 = Female)
    confidence: float (0-1)
    """
```

## Quick Test

```bash
python inference.py path/to/image.jpg
```

The script prints:

```text
label=<0_or_1> confidence=<0_to_1>
```

## Packaging
1. Rename this folder from `TeamID_TeamName` to your actual team name format:
   `ActualTeamID_ActualTeamName`
2. Zip the folder as:
   `ActualTeamID_ActualTeamName_GenderClassification.zip`

Ensure this final ZIP root directly contains:
- `model/model.pth`
- `inference.py`
- `requirements.txt`
- `model_card.pdf`
- `README.md`
