import torch
from .const import INSTRUCTION, inference_template


def inference(input_str, model, tokenizer, device):
    input_str = inference_template.format(instruction=INSTRUCTION, input=input_str)
    inp_encoded = tokenizer(input_str, return_tensors="pt")
    X, a = inp_encoded["input_ids"].to(device), inp_encoded["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a, max_length=256)
    decoded_output = tokenizer.decode(output[0])
    return decoded_output
