import torch

def inference(question, model, tokenizer, device, instruction):
    infer_template = """<startofstring> <Instruction>
{instruction}

<Question>
{input}

<Answer>"""

    input_str = infer_template.format(
        instruction=instruction,
        input=question,
    )
    input_data = tokenizer(input_str, return_tensors="pt")
    X = input_data["input_ids"].to(device)
    a = input_data["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a, max_length=256)
    output = tokenizer.decode(output[0])
    return output
