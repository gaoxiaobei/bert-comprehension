import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Solve RC using DeBERTa NLI logic.")
    parser.add_argument("--article", default="article.md", help="Path to the article file.")
    parser.add_argument("--qa", default="QA.csv", help="Path to the QA CSV file.")
    # We use the NLI-tuned version of DeBERTa-v3-large for general reasoning
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-large", help="NLI Model name.")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum allowed context window.")
    
    args = parser.parse_args()

    if not os.path.exists(args.article) or not os.path.exists(args.qa):
        print("Error: Files not found.")
        return

    with open(args.article, "r", encoding="utf-8") as f:
        article = f.read().strip()
    df = pd.read_csv(args.qa)

    print(f"Loading NLI Model: {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # NLI models use Sequence Classification
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    reverse_option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    correct_count = 0

    for index, row in df.iterrows():
        question = row['question']
        options = [row['choice A'], row['choice B'], row['choice C'], row['choice D']]
        correct_label = row['correct answer'].strip()
        
        print(f"\n=== Question {index + 1} ===")
        print(f"  [Q]: {question}")

        scores = []
        for opt in options:
            # NLI logic: Premise = Article, Hypothesis = Question + Answer
            # We check how much the Article "Entails" the Answer
            hypothesis = f"{question} {opt}"
            
            inputs = tokenizer(
                article, 
                hypothesis,
                max_length=args.max_len,
                truncation="only_first",
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # In cross-encoder/nli-deberta-v3-large:
                # index 0 = contradiction, 1 = neutral, 2 = entailment
                # We care about the 'entailment' score
                entail_logit = outputs.logits[0][2].item()
                scores.append(entail_logit)

        # Convert logits to probabilities for display
        probs = torch.softmax(torch.tensor(scores), dim=0).tolist()
        predicted_idx = torch.tensor(scores).argmax().item()
        predicted_label = reverse_option_map[predicted_idx]

        for i, prob in enumerate(probs):
            tag = " <--" if i == predicted_idx else ""
            print(f"    {reverse_option_map[i]}: {prob:.4f} (Logit: {scores[i]:.2f}){tag}")
        
        print(f"  Result: Predicted [{predicted_label}] | Correct [{correct_label}]", end=" ")
        if predicted_label == correct_label:
            print("✅")
            correct_count += 1
        else:
            print("❌")

    print(f"\nFinal Accuracy: {correct_count}/{len(df)} ({(correct_count/len(df))*100:.2f}%)")

if __name__ == "__main__":
    main()