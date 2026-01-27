import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Solve reading comprehension problems using DeBERTa with Sliding Window.")
    parser.add_argument("--article", default="article.md", help="Path to the article file.")
    parser.add_argument("--qa", default="QA.csv", help="Path to the QA CSV file.")
    parser.add_argument("--model", default="artianand/deberta-v3-large-race", help="Hugging Face model name.")
    
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
        options = [str(row['choice A']), str(row['choice B']), str(row['choice C']), str(row['choice D'])]
        correct_label = row['correct answer'].strip()
        
        print(f"Question {index + 1}: {question}")

        # 1. Prepare Inputs
        # We pair the Question with the Article. 
        # Note: Some fine-tuned models prefer (Article, Question+Option), others (Question, Article).
        # Standard huggingface MC implementation often uses [[context, question+opt]] pairs.
        # Here we follow the logic: Context=Article (Long), Candidate=Question+Option (Short).
        
        # We need to construct pairs for the tokenizer:
        # Pair 1: (Question + Option A, Article)
        # Pair 2: (Question + Option B, Article) ...
        # We put Article second so we can truncate it using "only_second".
        
        prompts = [f"{question} {opt}" for opt in options]
        contexts = [article] * 4

        # 2. Tokenize with Sliding Window
        inputs = tokenizer(
            prompts,
            contexts,
            max_length=512,
            truncation="only_second", # Truncate the Article, keep Question+Option intact
            stride=128,               # Overlap between windows
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length",
            return_tensors="pt"
        )

        # 3. Reshape Inputs for MultipleChoice Model
        # The tokenizer returns a flat list of all chunks for all options.
        # We need to restructure this into batches of 4 options per window.
        
        # 'overflow_to_sample_mapping' tells us which original option (0,1,2,3) a chunk belongs to.
        sample_map = inputs.pop("overflow_to_sample_mapping")
        
        # Find out how many chunks we generated per option.
        # Ideally, since the Article is the same, each option should have the same number of chunks.
        # However, if one option text is significantly longer, it might spawn an extra chunk.
        # We take the minimum chunk count to ensure we can form complete batches of 4.
        
        # Count chunks per option index (0 to 3)
        chunk_counts = torch.bincount(sample_map, minlength=4)
        num_windows = chunk_counts.min().item()
        
        # Prepare storage for the reshaped tensors
        # Shape needed: [Num_Windows, 4 (Choices), Seq_Len]
        reshaped_input_ids = []
        reshaped_attention = []
        reshaped_token_type = [] # For BERT/DeBERTa
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs.get("token_type_ids", None)

        # We assume the chunks for each option come in order.
        # We will extract the first 'num_windows' chunks for each option.
        for window_i in range(num_windows):
            window_input_ids = []
            window_attention = []
            window_token_type = []
            
            for option_idx in range(4):
                # Find the indices in the flat list corresponding to this option
                option_chunk_indices = (sample_map == option_idx).nonzero(as_tuple=True)[0]
                
                # Get the specific chunk for this window
                chunk_idx = option_chunk_indices[window_i]
                
                window_input_ids.append(input_ids[chunk_idx])
                window_attention.append(attention_mask[chunk_idx])
                if token_type_ids is not None:
                    window_token_type.append(token_type_ids[chunk_idx])
            
            reshaped_input_ids.append(torch.stack(window_input_ids))
            reshaped_attention.append(torch.stack(window_attention))
            if token_type_ids is not None:
                reshaped_token_type.append(torch.stack(window_token_type))

        # Stack into tensors: [Num_Windows, 4, Seq_Len]
        batch_input_ids = torch.stack(reshaped_input_ids).to(device)
        batch_attention = torch.stack(reshaped_attention).to(device)
        batch_token_type = torch.stack(reshaped_token_type).to(device) if token_type_ids is not None else None

        # 4. Inference loop over windows
        all_logits = []

        with torch.no_grad():
            # Iterate over the windows (chunks)
            for i in range(num_windows):
                # Slice the batch: [1, 4, Seq_Len] - Simulating batch size 1
                b_input_ids = batch_input_ids[i].unsqueeze(0)
                b_att_mask = batch_attention[i].unsqueeze(0)
                b_token_type = batch_token_type[i].unsqueeze(0) if batch_token_type is not None else None

                outputs = model(
                    input_ids=b_input_ids, 
                    attention_mask=b_att_mask, 
                    token_type_ids=b_token_type
                )
                
                # outputs.logits shape: [1, 4]
                all_logits.append(outputs.logits.squeeze(0)) # Store [4]

        # 5. Aggregate Results
        # Stack logits: [Num_Windows, 4]
        stacked_logits = torch.stack(all_logits)
        
        # Strategy: Average the logits across all windows
        avg_logits = torch.mean(stacked_logits, dim=0)
        probs = torch.softmax(avg_logits, dim=0).cpu().tolist()

        predicted_class_id = torch.argmax(avg_logits).item()
        predicted_label = reverse_option_map[predicted_class_id]
        
        print(f"  Probabilities (Avg across {num_windows} windows):")
        for i, prob in enumerate(probs):
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