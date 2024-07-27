import torch
import tqdm
from .inference import inference

def train(chatData, model, optim, device, tokenizer):
    epochs = 1

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        for X, a in tqdm.tqdm(chatData):
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            loss = model(X, attention_mask=a, labels=X).loss
            loss.backward()
            optim.step()
        torch.save(model.state_dict(), "model_state.pt")
        print(inference("Could you verify the residence of Alex Smith?", model, tokenizer, device))
