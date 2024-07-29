from torch.utils.data import Dataset
from datasets import load_dataset
from .const import template

# Dataset Section
class PrivacyDataset(Dataset):
    def __init__(self, tokenizer, split="train"):
        self.data = load_dataset("Yijia-Xiao/PPLM-PQA", split='train')
        self.X = [
            template.format(
                instruction=sample['instruction'],
                input=sample['input'],
                output=sample['output'],
                cleaned_output=sample['cleaned_output']
            ) for sample in self.data
        ]
        # Use 70% for training and 30% for validation
        if split == "train":
            self.X = self.X[:int(0.7 * len(self.X))]
            self.data = self.data[:int(0.7 * len(self.data))]
        elif split == "eval":
            self.X = self.X[int(0.7 * len(self.X)):]
            self.data = self.data[int(0.7 * len(self.data)):]
        else:
            raise ValueError("Invalid split. Please choose 'train' or 'eval'")
        
        self.X_encoded = tokenizer(self.X, max_length=64, truncation=True, padding="max_length", return_tensors="pt")
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx]
