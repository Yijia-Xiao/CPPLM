
# CPPLM: Large Language Models Can Be Contextual Privacy Protection Learners

CPPLM is a project that focuses on conditional language modeling using pretrained language models. This repository contains the implementation of the model, training scripts, and utilities necessary to run experiments.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

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

The main script can be used for training, inference, and evaluation. The mode can be specified using the `--mode` argument.

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

### Evaluation

To evaluate the model, ensure the model has been trained and the `model_state.pt` file is available. Then run:

```bash
python main.py --mode eval
```

## Project Structure

The repository is structured as follows:

```
.
├── README.md
├── cpplm
│   ├── __init__.py
│   ├── const.py
│   ├── dataset.py          # Defines the `PrivacyDataset` class for loading and processing the dataset
│   ├── inference.py        # Contains the `inference` function for generating responses
│   ├── model.py            # Contains the `get_model_and_tokenizer` function for initializing the model and tokenizer
│   ├── train.py            # Contains the `train` function for training the model
│   └── utils.py            # Contains utility functions
├── main.py                 # Entry point for training, inference, and evaluation
├── requirements.txt        # Required Python packages
└── utils
    ├── figures.py          # Utility functions for plotting figures
    └── scorer.py           # Utility functions for scoring model outputs
```

## Examples

Here are some examples of how to use CPPLM:

### Training Example

```bash
python main.py --mode train
```

### Inference Example

```bash
python main.py --mode inference
```

### Evaluation Example

```bash
python main.py --mode eval
```

## Contributing

We welcome contributions to CPPLM. If you would like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
