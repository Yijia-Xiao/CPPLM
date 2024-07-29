from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_model_and_tokenizer():
    # Tokenizer and model setup
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({
        "pad_token": "<pad>",
        "bos_token": "<startofstring>",
        "eos_token": "<endofstring>",
    })
    tokenizer.add_tokens([
        "<Instruction>", "<Question>", "<Answer>", "<Response>",
        "[Email]", "[Address]", "[Name]", "[SSH]"
    ])

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer
