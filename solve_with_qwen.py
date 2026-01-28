import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import argparse

def get_log_likelihood(model, tokenizer, context, target, device):
    """
    Calculates the mathematical probability of 'target' following 'context'.
    """
    # Concatenate context and target
    full_text = context + " " + target
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    target_ids = tokenizer(target, return_tensors="pt").input_ids.to(device)
    
    target_len = target_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(inputs.input_ids)
        logits = outputs.logits # Shape: [1, seq_len, vocab_size]

    # Shift logits and labels to align: prediction at index i is for label at i+1
    # We only care about the logits for the 'target' part of the sequence
    relevant_logits = logits[0, -(target_len + 1):-1, :]
    relevant_labels = target_ids[0]

    # Calculate cross-entropy (log-likelihood) for each token in the answer
    # This quantifies how 'surprised' the model is by the choice
    loss = F.cross_entropy(relevant_logits, relevant_labels, reduction='none')
    
    # Return the mean log-probability (negative of loss)
    return -loss.mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", default="cet-6-202506.md")
    parser.add_argument("--qa", default="cet6-2506.csv")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto")
    model.eval()

    with open(args.article, "r", encoding="utf-8") as f:
        article = f.read().strip()
    df = pd.read_csv(args.qa)

    correct_count = 0
    reverse_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for idx, row in df.iterrows():
        question = row['question']
        options = [row['choice A'], row['choice B'], row['choice C'], row['choice D']]
        correct = row['correct answer'].strip()

        # Context is the article and the question
        context_str = f"Article: {article}\n\nQuestion: {question}\nAnswer:"

        scores = []
        for opt in options:
            # We quantify the likelihood of this specific option string
            score = get_log_likelihood(model, tokenizer, context_str, opt, device)
            scores.append(score)

        # Convert log-likelihoods to probabilities for visualization (Softmax)
        probs = F.softmax(torch.tensor(scores), dim=0).tolist()
        pred_idx = torch.tensor(scores).argmax().item()
        pred_label = reverse_map[pred_idx]

        print(f"\nQ{idx+1}: {question}")
        for i, p in enumerate(probs):
            tag = " <--" if i == pred_idx else ""
            print(f"  {reverse_map[i]}: {p:.4f} (Log-L: {scores[i]:.4f}){tag}")
        
        print(f"Result: {pred_label} | Correct: {correct}", "✅" if pred_label == correct else "❌")
        if pred_label == correct: correct_count += 1

    print(f"\nAccuracy: {correct_count}/{len(df)} ({(correct_count/len(df))*100:.2f}%)")

if __name__ == "__main__":
    main()