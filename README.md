Here is the corrected, "un-broken" README.md:

Markdown
# Detecting Hallucinations via Internal State Conflict

This repository implements a hallucination detector for GPT-2 using Logit Lens KL-divergence features.

## ⚠️ Critical Setup Requirement
For the imports to work, you MUST maintain the following directory structure. The detector repo and the original BackwardLens repo must sit side-by-side in the same parent folder:

```text
NLP/
├── BackwardLens/                 # Clone original repo here
└── backward-lens-hallucination/  # This repo
    ├── main.py
    ├── requirements.txt
    └── hallucinations_detector/   # Core logic folder
Installation
Clone both repositories into a single parent folder:

Bash
git clone [https://github.com/itamarassaf-TAU/backward-lens-hallucination.git](https://github.com/itamarassaf-TAU/backward-lens-hallucination.git)
git clone [https://github.com/KritzR/BackwardLens.git](https://github.com/KritzR/BackwardLens.git)
Install dependencies:

Bash
cd backward-lens-hallucination
pip install -r requirements.txt
Running the Code
Generate Features: Run the main script from the backward-lens-hallucination root:

Bash
python main.py
Train & Evaluate: Run the classifier script:

Bash
python hallucinations_detector/nn_classifier.py
Authors
Itamar Assaf

Nadav Orenstein
