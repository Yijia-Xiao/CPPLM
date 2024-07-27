import torch

def inference(inp, model, tokenizer, device, instruction):
    infer_template = """<startofstring> <Instruction>
{instruction}

<Question>
{input}

<Answer>"""

    inp = infer_template.format(
        instruction=instruction,
        input=inp,
    )
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a, max_length=256)
    output = tokenizer.decode(output[0])
    return output
