import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Solve reading comprehension problems using DeBERTa.")
    parser.add_argument("--article", default="article.md", help="Path to the article file.")
    parser.add_argument("--qa", default="QA.csv", help="Path to the QA CSV file.")
    parser.add_argument("--model", default="artianand/deberta-v3-large-race", help="Hugging Face model name.")
    
    args = parser.parse_args()

    # File paths
    article_path = args.article
    qa_path = args.qa
    model_name = args.model

    if not os.path.exists(article_path):
        print(f"Error: Article file '{article_path}' not found.")
        return
    if not os.path.exists(qa_path):
        print(f"Error: QA file '{qa_path}' not found.")
        return

    # Read article
    with open(article_path, "r", encoding="utf-8") as f:
        article = f.read().strip()

    # Read QA
    df = pd.read_csv(qa_path)

    # Load model and tokenizer
    print(f"Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Map for options
    option_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    reverse_option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    print("\nStarting comprehension task...\n")

    correct_count = 0
    total_questions = len(df)

    for index, row in df.iterrows():
        question = row['question']
        options = [row['choice A'], row['choice B'], row['choice C'], row['choice D']]
        correct_label = row['correct answer'].strip()
        
        print(f"Question {index + 1}: {question}")
        
        # Prepare inputs
        # Context is the article.
        # Candidates are Question + Option
        
        contexts = [article] * 4
        candidates = [f"{question} {option}" for option in options]
        
        # Sliding window tokenization
        # Text A: Question (short), Text B: Article (long)
        # DeBERTa tokenizer also supports this.
        
        tokens = tokenizer(
            [question] * 4, 
            [article] * 4,
            max_length=512,
            truncation="only_second",
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length",
            return_tensors="pt"
        )
        
        sample_map = tokens.pop("overflow_to_sample_mapping")
        num_chunks = len(tokens['input_ids']) // 4
        
        chunk_batches = []
        for i in range(num_chunks):
            # Option 0's i-th chunk is at index: i
            # Option 1's i-th chunk is at index: i + num_chunks (assuming consistent chunking)
             indices = [i + j * num_chunks for j in range(4)]
            
             batch_inputs = {
                k: v[indices] for k, v in tokens.items() 
                if isinstance(v, torch.Tensor)
            }
             chunk_batches.append(batch_inputs)

        model.eval()
        chunk_probs_list = []

        with torch.no_grad():
            for batch in chunk_batches:
                b_input_ids = batch['input_ids'].unsqueeze(0)
                b_att_mask = batch['attention_mask'].unsqueeze(0)
                b_token_type = batch['token_type_ids'].unsqueeze(0) if 'token_type_ids' in batch else None
                
                outputs = model(input_ids=b_input_ids, attention_mask=b_att_mask, token_type_ids=b_token_type)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).squeeze(0)
                chunk_probs_list.append(probs)
        
        # Average probabilities
        avg_probs = torch.stack(chunk_probs_list).mean(dim=0)
        final_probs = avg_probs.tolist()

        predicted_class_id = avg_probs.argmax().item()
        predicted_label = reverse_option_map[predicted_class_id]
        
        print(f"  Probabilities (Averaged across {num_chunks} chunks):")
        for i, prob in enumerate(final_probs):
            print(f"    {reverse_option_map[i]}: {prob:.4f}")
        
        print(f"  Predicted: {predicted_label}")
        print(f"  Correct:   {correct_label}")
        
        if predicted_label == correct_label:
            print("  Result:    CORRECT")
            correct_count += 1
        else:
            print("  Result:    INCORRECT")
        print("-" * 30)

    print(f"\nFinished. Accuracy: {correct_count}/{total_questions} ({correct_count/total_questions*100:.2f}%)")

if __name__ == "__main__":
    main()
