# Detecting Hallucinations via Internal State Conflict

This repository implements a hallucination detector for GPT-2 using Backward Lens KL-divergence features.

## Requirements

- Python `>=3.10,<3.13` (Python 3.13 is not supported by this dependency stack yet)
- `pip`
- `venv`

## Required Directory Layout

`BackwardLens` must be a sibling directory of this repository:

```text
NLP/
├── BackwardLens/
└── backward-lens-hallucination/
    ├── requirements.txt
    └── hallucinations_detector/
```

## Installation

```bash
git clone https://github.com/shacharKZ/BackwardLens
git clone https://github.com/itamarassaf-TAU/backward-lens-hallucination.git

# Use Python 3.10/3.11/3.12
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backward-lens-hallucination/requirements.txt
```

## Run

From repository root:

```bash
PYTHONPATH=BackwardLens python backward-lens-hallucination/hallucinations_detector/main.py --dataset facts
```

## Notes

- `hallucinations_detector/nn_classifier.py` is a module used by `main.py`, not a standalone entrypoint.
- The first run downloads the model from Hugging Face unless already cached.

## Authors

- Itamar Assaf
- Nadav Orenstein
