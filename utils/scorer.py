import json
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# Load the provided JSON file
file_path = 'parsed_example.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Function to calculate scores
def calculate_scores(entry):
    output = entry[0]['output']
    response = entry[-1]

    # Calculate BERTScore
    P, R, F1 = bert_score([response], [output], lang="en", verbose=True)
    bert_scores = {
        "precision": P.mean().item(),
        "recall": R.mean().item(),
        "f1": F1.mean().item()
    }

    # Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(output, response)
    rouge_scores = {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure
    }

    return bert_scores, rouge_scores

# Calculate scores for each entry and append to the entry
for entry in data:
    bert_scores, rouge_scores = calculate_scores(entry)
    entry.append({
        "bert_score": bert_scores,
        "rouge_scores": rouge_scores
    })

# Save the modified data with scores to a new JSON file
output_path = './scored_example.json'
with open(output_path, 'w') as outfile:
    json.dump(data, outfile, indent=4)
