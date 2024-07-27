from torch.utils.data import Dataset
from datasets import load_dataset

class PrivacyDataset(Dataset):
    def __init__(self, tokenizer, split="train"):
        self.template = """<startofstring> <Instruction>
{instruction}

<Question>
{input}

<Answer>
{output}

<Response>
{cleaned_output}"""
        self.data = load_dataset("Yijia-Xiao/PPLM-PQA", split=split)
        self.X = [self.template.format(
                    instruction=sample['instruction'],
                    input=sample['input'],
                    output=sample['output'],
                    cleaned_output=sample['cleaned_output']
                ) for sample in self.data]

        self.X_encoded = tokenizer(self.X, max_length=192, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])
