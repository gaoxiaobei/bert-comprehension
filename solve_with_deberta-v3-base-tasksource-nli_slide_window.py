import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

def split_into_overlapping_chunks(text, tokenizer, max_tokens=450, overlap=100):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))
        if i + max_tokens >= len(tokens): break
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", default="cet-6-202506.md")
    parser.add_argument("--qa", default="cet6-2506.csv")
    parser.add_argument("--model", default="sileod/deberta-v3-base-tasksource-nli")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    # Label indices for tasksource-nli
    label_map = model.config.label2id
    ent_idx = label_map.get('entailment', 0)
    con_idx = label_map.get('contradiction', 2)

    with open(args.article, "r", encoding="utf-8") as f:
        article_text = f.read().strip()
    df = pd.read_csv(args.qa)

    # 1. Break article into small, digestible chunks
    chunks = split_into_overlapping_chunks(article_text, tokenizer)
    print(f"Article split into {len(chunks)} chunks for full-text scanning.")

    correct_count = 0
    reverse_map = {0:'A', 1:'B', 2:'C', 3:'D'}

    for idx, row in df.iterrows():
        question = row['question']
        options = [row['choice A'], row['choice B'], row['choice C'], row['choice D']]
        correct = row['correct answer'].strip()

        choice_scores = []
        for opt in options:
            hypothesis = f"{question} {opt}"
            
            # 2. Scan every chunk and find the HIGHEST support for this option
            best_chunk_score = -999
            for chunk in chunks:
                inputs = tokenizer(chunk, hypothesis, truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits[0]
                    # Score = Entailment - Contradiction
                    score = logits[ent_idx].item() - logits[con_idx].item()
                    if score > best_chunk_score:
                        best_chunk_score = score
            
            choice_scores.append(best_chunk_score)

        pred_idx = torch.tensor(choice_scores).argmax().item()
        pred_label = reverse_map[pred_idx]

        print(f"\nQ{idx+1}: {question}")
        print(f"  Result: [{pred_label}] | Correct: [{correct}]", "✅" if pred_label==correct else "❌")
        if pred_label == correct: correct_count += 1

    print(f"\nFinal Accuracy: {correct_count}/{len(df)} ({(correct_count/len(df))*100:.2f}%)")

if __name__ == "__main__":
    main()