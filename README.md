# CPPLM

A repository for training and inference with a GPT-2 model on the PPLM-PQA dataset.

## Installation

Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Training

To train the model, set `TRAIN = True` in `main.py` and run:
```bash
python main.py
```

## Inference

To run inference with the trained model, set `TRAIN = False` in `main.py` and run:
```bash
python main.py
```

## Dataset

The dataset used is [PPLM-PQA](https://huggingface.co/datasets/Yijia-Xiao/PPLM-PQA).

## Files

- `cpplm/`: Contains the main code for data processing, model training, and inference.
- `utils/`: Contains utility functions.
- `main.py`: Entry point for training and inference.
- `README.md`: This file.
- `requirements.txt`: Required Python packages.

## Requirements

- torch
- transformers
- datasets
- tqdm
