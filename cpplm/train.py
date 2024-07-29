import torch
import tqdm
from .inference import inference
from .utils import parse_output
from .const import NUM_EPOCH

def train(priv_dataset, model, optim, device, tokenizer):
    for epoch in range(NUM_EPOCH):
        for X, a in tqdm.tqdm(priv_dataset):
            X, a = X.to(device), a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), f"./data/model_state_{epoch}.pt")
        print(parse_output(inference("Could you verify the residence of Alex Smith?", model, tokenizer, device)))
