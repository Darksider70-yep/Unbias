# Gender Classification Submission Project

This repository has been updated for the event task:
- Input: facial image
- Output: `0` (Male) or `1` (Female)
- Runtime: Python 3.10, CPU only, offline
- Model format: PyTorch `.pth` only

## Submission Template

Use the folder:

- `TeamID_TeamName/`

It already follows the required structure:

```text
TeamID_TeamName/
|-- model/
|   `-- model.pth
|-- inference.py
|-- requirements.txt
|-- model_card.pdf
`-- README.md
```

`inference.py` includes the required function:

```python
def predict(image_path):
    """
    Returns:
    label: int (0 = Male, 1 = Female)
    confidence: float (0-1)
    """
```

## Build Final ZIP

Run:

```bash
python scripts/build_submission_zip.py --team-id <YOUR_TEAM_ID> --team-name <YOUR_TEAM_NAME>
```

This generates:

- `<TEAM_ID>_<TEAM_NAME>/`
- `<TEAM_ID>_<TEAM_NAME>_GenderClassification.zip`

## Notes

- Replace the baseline `model/model.pth` with your final trained weights before submission.
- Update `model_card.pdf` with your real dataset details, training setup, and bias analysis.
