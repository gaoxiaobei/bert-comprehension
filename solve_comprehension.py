import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Solve reading comprehension problems using ALBERT with Sliding Window.")
    parser.add_argument("--article", default="article.md", help="Path to the article file (markdown or text).")
    parser.add_argument("--qa", default="QA.csv", help="Path to the QA CSV file.")
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
        # Convert all choices to strings to avoid type errors
        options = [str(row['choice A']), str(row['choice B']), str(row['choice C']), str(row['choice D'])]
        correct_label = row['correct answer'].strip()
        
        print(f"Question {index + 1}: {question}")
        
        # 1. Prepare Inputs
        # We want the input to be: [CLS] Question + Option [SEP] Article [SEP]
        # We truncate "only_second" so the Article slides, but Question+Option remains complete.
        
        prompts = [f"{question} {opt}" for opt in options]
        contexts = [article] * 4
        
        # 2. Tokenize with Sliding Window
        inputs = tokenizer(
            prompts,       # Text A (Keep intact)
            contexts,      # Text B (Slide over this)
            max_length=512,
            truncation="only_second", 
            stride=128,    # Amount of overlap between windows
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 3. Reshape/Regroup Inputs
        # The tokenizer returns a flat list. We need to organize it into batches where
        # each batch contains the 4 options for a specific "window" of the text.
        
        sample_map = inputs.pop("overflow_to_sample_mapping") 
        # sample_map is a tensor like [0, 0, 1, 1, 2, 2, 3, 3] indicating which option (0-3) a chunk belongs to.

        # Calculate how many chunks exist for each option. 
        # (They should be equal since the context is identical, but we calculate min for safety).
        chunk_counts = torch.bincount(sample_map, minlength=4)
        num_windows = chunk_counts.min().item()

        # Extract tensor data
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs.get("token_type_ids", None) # ALBERT needs this

        chunk_logits_list = []

        with torch.no_grad():
            # Iterate through each window position (e.g., Window 0, Window 1...)
            for i in range(num_windows):
                
                # Build the batch for this specific window
                window_input_ids = []
                window_att_mask = []
                window_token_types = []

                for option_idx in range(4):
                    # Find the index of the i-th chunk for the current option_idx
                    # (sample_map == option_idx) gives indices for that option
                    # .nonzero() returns the actual positions in the flat list
                    indices = (sample_map == option_idx).nonzero(as_tuple=True)[0]
                    chunk_idx = indices[i]

                    window_input_ids.append(input_ids[chunk_idx])
                    window_att_mask.append(attention_mask[chunk_idx])
                    if token_type_ids is not None:
                        window_token_types.append(token_type_ids[chunk_idx])

                # Stack to create batch of shape [1, 4, seq_len]
                b_input_ids = torch.stack(window_input_ids).unsqueeze(0).to(device)
                b_att_mask = torch.stack(window_att_mask).unsqueeze(0).to(device)
                b_token_type = torch.stack(window_token_types).unsqueeze(0).to(device) if token_type_ids is not None else None
                
                # Model Inference
                outputs = model(
                    input_ids=b_input_ids, 
                    attention_mask=b_att_mask, 
                    token_type_ids=b_token_type
                )
                
                # Store logits for this window: Shape [4]
                chunk_logits_list.append(outputs.logits.squeeze(0))

        # 4. Aggregate Results
        # Stack all window logits: [num_windows, 4]
        all_logits = torch.stack(chunk_logits_list)
        
        # Method: Mean Pooling of Logits (Averaging scores across all text windows)
        avg_logits = torch.mean(all_logits, dim=0)
        
        # Calculate final probabilities
        final_probs = torch.softmax(avg_logits, dim=0).cpu().tolist()
        predicted_class_id = torch.argmax(avg_logits).item()
        predicted_label = reverse_option_map[predicted_class_id]
        
        print(f"  Probabilities (Averaged across {num_windows} windows):")
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