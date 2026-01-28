import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", default="cet-6-202506.md")
    parser.add_argument("--qa", default="cet6-2506.csv")
    # This is currently the highest-rated zero-shot NLI model on HuggingFace
    parser.add_argument("--model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading High-Power Model: {args.model}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    # Label Mapping for MoritzLaurer model: 0: entailment, 1: neutral, 2: contradiction
    ent_idx = 0 
    con_idx = 2

    with open(args.article, "r", encoding="utf-8") as f:
        article = f.read().strip()
    df = pd.read_csv(args.qa)

    correct_count = 0
    reverse_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    for idx, row in df.iterrows():
        question = row['question']
        options = [row['choice A'], row['choice B'], row['choice C'], row['choice D']]
        correct = row['correct answer'].strip()

        scores = []
        for opt in options:
            # We structure the prompt clearly
            hypothesis = f"{question} {opt}"
            
            # Use max possible window (1024) to keep global context
            inputs = tokenizer(article, hypothesis, truncation="only_first", max_length=1024, return_tensors="pt").to(device)
            
            with torch.no_grad():
                logits = model(**inputs).logits[0]
                # Logic: We want high entailment and LOW contradiction
                # We weight entailment heavily
                score = logits[ent_idx].item() - (logits[con_idx].item() * 0.5)
                scores.append(score)

        pred_idx = torch.tensor(scores).argmax().item()
        pred_label = reverse_map[pred_idx]

        print(f"\nQ{idx+1}: {question}")
        for i, s in enumerate(scores):
            tag = " <--" if i == pred_idx else ""
            print(f"  {reverse_map[i]}: {s:.2f}{tag}")
            
        print(f"Result: {pred_label} | Correct: {correct}", "✅" if pred_label == correct else "❌")
        if pred_label == correct: correct_count += 1

    print(f"\nFinal Accuracy: {correct_count}/{len(df)} ({(correct_count/len(df))*100:.2f}%)")

if __name__ == "__main__":
    main()