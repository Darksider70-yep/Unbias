# Gender Classification Submission Project

This repository is prepared for the event task:
- Input: one face image
- Output: `0` (Male) or `1` (Female)
- Runtime: Python 3.10, CPU only, offline
- Model file format: PyTorch `.pth`

## Final Submission Folder

Use:

- `2005_Greeninja/`

Structure:

```text
2005_Greeninja/
|-- model/
|   `-- model.pth
|-- inference.py
|-- requirements.txt
|-- model_card.pdf
`-- README.md
```

## Run Inference

```bash
python -m pip install -r 2005_Greeninja/requirements.txt
python 2005_Greeninja/inference.py scripts/sample.jpg
```

## Build ZIP

Zip `2005_Greeninja/` as:

- `2005_Greeninja_GenderClassification.zip`
