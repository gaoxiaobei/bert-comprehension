import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

def get_sequence_logprob(model, tokenizer, context, target, device):
    """Calculates the mean log-probability of the target string given the context."""
    full_text = context + " " + target
    inputs = tokenizer(full_text, return_tensors="pt").to(device)
    target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    
    target_len = target_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(inputs.input_ids)
        logits = outputs.logits # [1, seq_len, vocab_size]

    # Align logits with target labels
    # Logits at index i predict token at i+1
    relevant_logits = logits[0, -(target_len + 1):-1, :]
    relevant_labels = target_ids[0]

    # Calculate log-probabilities
    log_probs = F.log_softmax(relevant_logits, dim=-1)
    target_log_probs = log_probs.gather(1, relevant_labels.unsqueeze(1)).squeeze(1)
    
    return target_log_probs.mean().item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", default="cet-6-202506.md")
    parser.add_argument("--qa", default="cet6-2506.csv")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B") # Latest Qwen3
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
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

        print(f"\nQ{idx+1}: {question}")
        
        pmi_scores = []
        for opt in options:
            # 1. P(Choice | Article + Question)
            context_with_article = f"Article: {article}\nQuestion: {question}\nAnswer:"
            lp_with = get_sequence_logprob(model, tokenizer, context_with_article, opt, device)
            
            # 2. P(Choice | Question) - the "Prior" bias
            context_without_article = f"Question: {question}\nAnswer:"
            lp_without = get_sequence_logprob(model, tokenizer, context_without_article, opt, device)
            
            # PMI = Info gained from Article
            pmi_scores.append(lp_with - lp_without)

        # Quantification via Softmax of PMI scores
        probs = F.softmax(torch.tensor(pmi_scores), dim=0).tolist()
        pred_idx = torch.tensor(pmi_scores).argmax().item()
        pred_label = reverse_map[pred_idx]

        for i, p in enumerate(probs):
            tag = " <--" if i == pred_idx else ""
            print(f"  {reverse_map[i]}: {p:.4f} (PMI: {pmi_scores[i]:.4f}){tag}")
        
        print(f"Result: {pred_label} | Correct: {correct}", "✅" if pred_label == correct else "❌")
        if pred_label == correct: correct_count += 1

    print(f"\nFinal Accuracy: {correct_count}/{len(df)} ({(correct_count/len(df))*100:.2f}%)")

if __name__ == "__main__":
    main()