import torch
from torch.utils.data import DataLoader
from transformers import AdamW

from cpplm.dataset import PrivacyDataset
from cpplm.model import get_model_and_tokenizer
from cpplm.train import train
from cpplm.inference import inference
from utils.utils import print_output
import argparse

parser = argparse.ArgumentParser(description='CPPLM')

parser.add_argument('--mode', type=str, default='train', help='train or inference')
mode = parser.parse_args().mode

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

model, tokenizer = get_model_and_tokenizer()
model = model.to(device)

chatData = PrivacyDataset(tokenizer)
chatData = DataLoader(chatData, batch_size=128, shuffle=True)

model.train()

optim = AdamW(model.parameters(), lr=1e-3)

instruction = "Answer this question truthfully"

if mode == 'train':
    print("Training CPPLM .... ")    
    train(chatData, model, optim, device, tokenizer)
    print(inference("Could you verify the residence of Alex Smith?", model, tokenizer, device, instruction))

else:
    print("Loading CPPLM model ...")
    model.load_state_dict(torch.load("model_state.pt"))
    model = model.to(device)
    model.eval()

print("Inference from CPPLM ...")
question = "Could you verify the residence of Alex Smith?"

print(print_output(inference(question, model, tokenizer, device, instruction)))

print("Enter a question or type 'exit' to quit")

while question != 'exit':
    question = input()
    print(print_output(inference(question, model, tokenizer, device, instruction)))
