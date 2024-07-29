import json
import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from cpplm.dataset import PrivacyDataset
from cpplm.model import get_model_and_tokenizer
from cpplm.train import train
from cpplm.inference import inference
from cpplm.utils import parse_output
from cpplm.const import NUM_EPOCH

import argparse


# Main Section
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CPPLM')
    parser.add_argument('--mode', type=str, default='train', help='train or inference')
    
    args = parser.parse_args()
    mode = args.mode
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    model, tokenizer = get_model_and_tokenizer()
    model = model.to(device)

    # Load dataset and DataLoader setup
    priv_dataset = PrivacyDataset(tokenizer)
    priv_dataset = DataLoader(priv_dataset, batch_size=512, shuffle=True)

    # Optimizer setup
    optim = AdamW(model.parameters(), lr=5e-4)
    
    # Training or Inference
    if mode == 'train':
        print("Training...")
        train(priv_dataset, model, optim, device, tokenizer)
    elif mode == 'inference':
        print("Inference...")
        model.load_state_dict(torch.load(f"./data/model_state_{NUM_EPOCH - 1}.pt"))
        while True:
            question = input("Please enter a question: ")
            response = inference(question, model, tokenizer, device).replace("<pad>", "").replace("<endofstring>", "").replace("<startofstring>", "")
            print("Response:", parse_output(response))
    elif mode == 'eval':
        print("Evaluation...")
        model.load_state_dict(torch.load(f"./data/model_state_{NUM_EPOCH - 1}.pt"))
        model.eval()
        priv_dataset = PrivacyDataset(tokenizer, split="eval")

        results = []
        with torch.no_grad():
            for sample_idx in tqdm.tqdm(range(len(priv_dataset.data['input']))):
                sample = {
                    'input': priv_dataset.data['input'][sample_idx],
                    'output': priv_dataset.data['output'][sample_idx],
                    'cleaned_output': priv_dataset.data['cleaned_output'][sample_idx]
                }
                response = inference(sample['input'], model, tokenizer, device).replace("<pad>", "").replace("<endofstring>", "").replace("<startofstring>", "")
                results.append([sample, parse_output(response)])

        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)

    else:
        raise ValueError("Invalid mode. Please choose 'train', 'inference' or 'eval'")
