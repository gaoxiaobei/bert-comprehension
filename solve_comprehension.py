import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Solve reading comprehension with dynamic aggregation strategies.")
    parser.add_argument("--article", default="article.md", help="Path to the article file.")
    parser.add_argument("--qa", default="QA.csv", help="Path to the QA CSV file with 'question type' column.")
    parser.add_argument("--model", default="Riiid/kda-albert-xxlarge-v2-race", help="Hugging Face model name.")
    
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    
    # Check if 'question type' column exists
    if 'question type' not in df.columns:
        print("Warning: 'question type' column not found in CSV. Defaulting to 'main idea' (Average) strategy.")

    # Load model and tokenizer
    print(f"Loading model: {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Map for options
    reverse_option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    print("\nStarting comprehension task...\n")

    correct_count = 0
    total_questions = len(df)

    for index, row in df.iterrows():
        question = row['question']
        options = [str(row['choice A']), str(row['choice B']), str(row['choice C']), str(row['choice D'])]
        correct_label = row['correct answer'].strip()
        
        # Determine Aggregation Strategy
        # Default to 'main idea' if column is missing or empty
        q_type_raw = row.get('question type', 'main idea')
        q_type = str(q_type_raw).strip().lower()
        
        print(f"Question {index + 1}: {question}")
        print(f"  Type:      {q_type_raw}")
        
        # 1. Prepare Inputs
        prompts = [f"{question} {opt}" for opt in options]
        contexts = [article] * 4
        
        # 2. Tokenize with Sliding Window
        inputs = tokenizer(
            prompts,
            contexts,
            max_length=512,
            truncation="only_second", 
            stride=128, 
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 3. Reshape/Regroup Inputs
        sample_map = inputs.pop("overflow_to_sample_mapping") 
        chunk_counts = torch.bincount(sample_map, minlength=4)
        num_windows = chunk_counts.min().item()

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs.get("token_type_ids", None)

        window_probs_list = []

        with torch.no_grad():
            for i in range(num_windows):
                # Build batch for the i-th window
                window_input_ids = []
                window_att_mask = []
                window_token_types = []

                for option_idx in range(4):
                    indices = (sample_map == option_idx).nonzero(as_tuple=True)[0]
                    chunk_idx = indices[i]

                    window_input_ids.append(input_ids[chunk_idx])
                    window_att_mask.append(attention_mask[chunk_idx])
                    if token_type_ids is not None:
                        window_token_types.append(token_type_ids[chunk_idx])

                b_input_ids = torch.stack(window_input_ids).unsqueeze(0).to(device)
                b_att_mask = torch.stack(window_att_mask).unsqueeze(0).to(device)
                b_token_type = torch.stack(window_token_types).unsqueeze(0).to(device) if token_type_ids is not None else None
                
                outputs = model(
                    input_ids=b_input_ids, 
                    attention_mask=b_att_mask, 
                    token_type_ids=b_token_type
                )
                
                # Calculate Probabilities for this specific window
                logits = outputs.logits.squeeze(0) # [4]
                probs = torch.softmax(logits, dim=0)
                window_probs_list.append(probs)

        # 4. Dynamic Aggregation
        # Stack: [num_windows, 4]
        all_probs_tensor = torch.stack(window_probs_list)
        
        final_scores = None
        strategy_desc = ""

        if q_type == 'detail':
            # Strategy: Max Pooling
            # Use the single highest probability found in any window for each option independently.
            final_scores, _ = torch.max(all_probs_tensor, dim=0)
            strategy_desc = "Max Pooling (Highest score across windows)"
        else:
            # Strategy: Mean Pooling (Default / Main Idea)
            # Average the probabilities across all windows.
            final_scores = torch.mean(all_probs_tensor, dim=0)
            strategy_desc = "Mean Pooling (Average score across windows)"

        # Get final prediction
        final_scores_list = final_scores.cpu().tolist()
        predicted_class_id = torch.argmax(final_scores).item()
        predicted_label = reverse_option_map[predicted_class_id]
        
        print(f"  Strategy:  {strategy_desc}")
        print(f"  Scores:")
        for i, score in enumerate(final_scores_list):
            print(f"    {reverse_option_map[i]}: {score:.4f}")
        
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