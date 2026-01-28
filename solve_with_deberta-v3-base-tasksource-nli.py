import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Solve RC using Tasksource NLI.")
    parser.add_argument("--article", default="article.md")
    parser.add_argument("--qa", default="QA.csv")
    parser.add_argument("--model", default="sileod/deberta-v3-base-tasksource-nli")
    parser.add_argument("--max_len", type=int, default=1024)
    args = parser.parse_args()

    if not os.path.exists(args.article) or not os.path.exists(args.qa):
        print("Error: Files not found.")
        return

    with open(args.article, "r", encoding="utf-8") as f:
        article = f.read().strip()
    df = pd.read_csv(args.qa)

    print(f"Loading Model: {args.model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    # Automatically identify which index is 'entailment'
    # Tasksource models usually use: 0: entailment, 1: neutral, 2: contradiction
    label_map = model.config.label2id
    entail_idx = label_map.get('entailment', label_map.get('ENTAILMENT', 0))
    contra_idx = label_map.get('contradiction', label_map.get('CONTRADICTION', 2))
    
    print(f"Detected Label Mapping: Entailment={entail_idx}, Contradiction={contra_idx}")

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
            # Hypothesis: "Question? Answer Choice"
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
                logits = outputs.logits[0]
                # We calculate a 'Net Support' score: Entailment minus Contradiction
                # This helps filter out 'distractors' that the model finds contradictory
                net_score = logits[entail_idx].item() - (logits[contra_idx].item() * 0.5)
                scores.append(net_score)

        probs = torch.softmax(torch.tensor(scores), dim=0).tolist()
        predicted_idx = torch.tensor(scores).argmax().item()
        predicted_label = reverse_option_map[predicted_idx]

        for i, prob in enumerate(probs):
            tag = " <--" if i == predicted_idx else ""
            print(f"    {reverse_option_map[i]}: {prob:.4f} (Net Score: {scores[i]:.2f}){tag}")
        
        print(f"  Result: Predicted [{predicted_label}] | Correct [{correct_label}]", end=" ")
        if predicted_label == correct_label:
            print("✅")
            correct_count += 1
        else:
            print("❌")

    print(f"\nFinal Accuracy: {correct_count}/{len(df)} ({(correct_count/len(df))*100:.2f}%)")

if __name__ == "__main__":
    main()