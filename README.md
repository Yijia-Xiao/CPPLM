
# CPPLM

A repository for training and inference with a GPT-2 model on the PPLM-PQA dataset.

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Requirements

- torch
- transformers
- datasets
- tqdm

## Dataset

The dataset used is [PPLM-PQA](https://huggingface.co/datasets/Yijia-Xiao/PPLM-PQA).

## Usage

The main script can be used for both training and inference. The mode can be specified using the `--mode` argument.

### Training

To train the model, run:

```bash
python main.py --mode train
```

### Inference

To run inference with the trained model, ensure the model has been trained and the `model_state.pt` file is available. Then run:

```bash
python main.py --mode inference
```

## Files

- `cpplm/`: Contains the main code for data processing, model training, and inference.
  - `dataset.py`: Defines the `PrivacyDataset` class for loading and processing the dataset.
  - `infer.py`: Contains the `infer` function for generating responses.
  - `model.py`: Contains the `get_model_and_tokenizer` function for initializing the model and tokenizer.
  - `train.py`: Contains the `train` function for training the model.
- `utils/`: Contains utility functions.
  - `utils.py`: Contains the `print_output` function for formatting the model output.
- `main.py`: Entry point for training and inference.
- `README.md`: This file.
- `requirements.txt`: Required Python packages.
