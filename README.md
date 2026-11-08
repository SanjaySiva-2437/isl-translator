# ISL Translator

Real-time Indian Sign Language (ISL) to text/speech translator.
SRM mini-project.

## Setup
python -m venv <path-outside-repo>
<path>\Scripts\Activate.ps1
pip install -r requirements.txt

## Extract landmarks
python scripts\extract_landmarks.py --videos <videos-dir> --out <landmarks-dir> --models <models-dir>

The landmark format is defined in docs/spec.md.

## Dataset
INCLUDE dataset (not included in this repo). Download it separately and keep it outside the repository.