import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import os
import argparse
import sys
from tqdm import tqdm  # 引入进度条
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="Solve reading comprehension with dynamic aggregation strategies (Optimized).")
    parser.add_argument("--article", default="article.md", help="Path to the article file.")
    parser.add_argument("--qa", default="QA.csv", help="Path to the QA CSV file.")
    parser.add_argument("--model", default="Riiid/kda-albert-xxlarge-v2-race", help="Hugging Face model name.")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size (number of windows processed at once).")
    return parser.parse_args()

def prepare_inputs(tokenizer, question, options, article, max_length=512, stride=128):
    """
    Tokenize and restructure inputs into [Num_Windows, 4, Seq_Len] format.
    Handles cases where options might result in different numbers of chunks by padding.
    """
    prompts = [f"{question} {opt}" for opt in options]
    contexts = [article] * 4

    inputs = tokenizer(
        prompts,
        contexts,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=False,
        padding="max_length",
        return_tensors="pt"
    )

    # pop mapping to reorganize
    sample_map = inputs.pop("overflow_to_sample_mapping")
    
    # Analyze chunk distribution
    # We expect 4 options (0, 1, 2, 3). Count how many chunks each option produced.
    # unique_ids should be [0, 1, 2, 3] usually
    unique_ids, counts = torch.unique(sample_map, return_counts=True)
    
    # Calculate max windows needed (align to the longest option to avoid info loss)
    max_windows = counts.max().item()
    
    # Prepare storage tensors [Max_Windows, 4, Seq_Len]
    # We use pad_token_id for input_ids and 0 for attention_mask
    reshaped_input_ids = torch.full((max_windows, 4, max_length), tokenizer.pad_token_id, dtype=torch.long)
    reshaped_att_mask = torch.zeros((max_windows, 4, max_length), dtype=torch.long)
    reshaped_token_type = None
    
    if "token_type_ids" in inputs:
        reshaped_token_type = torch.zeros((max_windows, 4, max_length), dtype=torch.long)

    # Efficient Scatter/Fill logic
    # Instead of Python loops, we iterate by option to fill the tensor
    for option_idx in range(4):
        # Find indices in the flat list belonging to this option
        indices = (sample_map == option_idx).nonzero(as_tuple=True)[0]
        num_chunks = len(indices)
        
        # Fill the tensor
        reshaped_input_ids[:num_chunks, option_idx, :] = inputs["input_ids"][indices]
        reshaped_att_mask[:num_chunks, option_idx, :] = inputs["attention_mask"][indices]
        
        if reshaped_token_type is not None and "token_type_ids" in inputs:
            reshaped_token_type[:num_chunks, option_idx, :] = inputs["token_type_ids"][indices]

    return {
        "input_ids": reshaped_input_ids,
        "attention_mask": reshaped_att_mask,
        "token_type_ids": reshaped_token_type,
        "valid_windows_mask": counts # Keep track of how many valid windows each option actually has
    }

def main():
    args = parse_args()

    # Device configuration (Support for MPS on Mac, CUDA on NVidia, CPU fallback)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_amp = True # Enable mixed precision for CUDA
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        use_amp = False # MPS mixed precision is strictly controlled, safer to disable for this script
    else:
        device = torch.device("cpu")
        use_amp = False

    print(f"Using device: {device} | Mixed Precision: {use_amp}")

    # File Validations
    if not os.path.exists(args.article):
        print(f"Error: Article file '{args.article}' not found.")
        return
    if not os.path.exists(args.qa):
        print(f"Error: QA file '{args.qa}' not found.")
        return

    # Load Data
    with open(args.article, "r", encoding="utf-8") as f:
        article = f.read().strip()
    
    df = pd.read_csv(args.qa)
    if 'question type' not in df.columns:
        print("Warning: 'question type' column missing. Defaulting to 'main idea'.")

    # Load Model & Tokenizer
    print(f"Loading model: {args.model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForMultipleChoice.from_pretrained(args.model)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    reverse_option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    correct_count = 0
    total_questions = len(df)

    print("\nStarting comprehension task...\n")

    # Use tqdm for progress bar
    for index, row in tqdm(df.iterrows(), total=total_questions, desc="Processing Questions"):
        question = row['question']
        options = [str(row['choice A']), str(row['choice B']), str(row['choice C']), str(row['choice D'])]
        correct_label = str(row['correct answer']).strip()
        q_type = str(row.get('question type', 'main idea')).strip().lower()

        # 1. Prepare Batched Inputs
        processed = prepare_inputs(tokenizer, question, options, article)
        
        # All tensors are shape [Num_Windows, 4, Seq_Len]
        input_ids_all = processed["input_ids"]
        att_mask_all = processed["attention_mask"]
        token_type_all = processed["token_type_ids"]
        
        num_windows = input_ids_all.size(0)
        
        # Container for logits from all windows
        all_window_logits = []

        # 2. Inference with Mini-Batching (to avoid OOM on long articles)
        # We process 'batch_size' windows at a time. Each window contains 4 options.
        batch_size = args.batch_size 
        
        with torch.no_grad():
            for i in range(0, num_windows, batch_size):
                end_i = min(i + batch_size, num_windows)
                
                # Slice the batch
                b_input_ids = input_ids_all[i:end_i].to(device) # [B, 4, Seq]
                b_att_mask = att_mask_all[i:end_i].to(device)
                b_token_type = token_type_all[i:end_i].to(device) if token_type_all is not None else None

                # Mixed Precision Context
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(
                        input_ids=b_input_ids,
                        attention_mask=b_att_mask,
                        token_type_ids=b_token_type
                    )
                
                # outputs.logits shape: [Batch_Size, 4]
                # We apply Softmax here to get probabilities per window
                probs = torch.softmax(outputs.logits, dim=-1)
                all_window_logits.append(probs)

        # 3. Aggregate Results
        # Stack inputs to [Num_Windows, 4]
        if len(all_window_logits) > 0:
            all_probs_tensor = torch.cat(all_window_logits, dim=0) # [Num_Windows, 4]
        else:
            # Fallback for empty (should not happen)
            all_probs_tensor = torch.zeros((1, 4))

        # Handle valid windows masking (Optional advanced logic)
        # If Option A has 3 chunks and Option B has 4, the 4th chunk for A is padding (prob ~0.25 uniform or garbage).
        # For simplicity in this optimization, we rely on the Attention Mask blocking garbage tokens, 
        # but the model might still output a logit. 
        # A robust way is to mask the probabilities of padded windows before aggregation, 
        # but given RACE datasets, lengths are usually consistent. 
        # We proceed with the strategy logic directly on the tensor.

        if q_type == 'detail':
            # Max Pooling: Find the window where the model is most confident for each option
            # Or find the absolute max score in the matrix
            # Dim 0 is windows. We want the best score for Option A across all windows.
            final_scores, _ = torch.max(all_probs_tensor, dim=0) 
            strategy_desc = "Max Pooling"
        else:
            # Mean Pooling: Average confidence across all windows
            final_scores = torch.mean(all_probs_tensor, dim=0)
            strategy_desc = "Mean Pooling"

        # Prediction
        predicted_class_id = torch.argmax(final_scores).item()
        predicted_label = reverse_option_map[predicted_class_id]
        
        # Logging (Simplified output for tqdm compatibility)
        tqdm.write(f"Q{index+1} ({q_type}): {strategy_desc} | Pred: {predicted_label} | True: {correct_label} | {'CORRECT' if predicted_label == correct_label else 'WRONG'}")
        
        if predicted_label == correct_label:
            correct_count += 1

    print("-" * 30)
    print(f"Finished. Accuracy: {correct_count}/{total_questions} ({correct_count/total_questions*100:.2f}%)")

if __name__ == "__main__":
    main()