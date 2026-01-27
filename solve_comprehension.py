import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Solve reading comprehension problems using ALBERT.")
    parser.add_argument("--article", default="article.md", help="Path to the article file (markdown or text).")
    parser.add_argument("--qa", default="QA.csv", help="Path to the QA CSV file.")
    parser.add_argument("--model", default="Riiid/kda-albert-xxlarge-v2-race", help="Hugging Face model name.")
    
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
    model_name = "Riiid/kda-albert-xxlarge-v2-race"
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
        
        # We need to repeat the context for each option
        contexts = [article] * 4
        candidates = [f"{question} {option}" for option in options]
        
        # Tokenize with sliding window
        # We process each choice + article pair. 
        # Since the article is long, we might get multiple chunks for each choice.
        # To keep it simple and effective: 
        # 1. Tokenize the article only first to get chunks? No, we need [CLS] Question [SEP] Article [SEP].
        #    Actually, standard BERT/ALBERT for RACE is [CLS] Question + Option [SEP] Article [SEP] or similar.
        #    Let's stick to the previous format: context=article, candidate=question+option.
        #    But we apply sliding window on the context.
        
        # We'll use a manual loop to handle the complexity of aligning chunks across 4 options.
        # We want to ensure that for a given "pass", all 4 options see the SAME chunk of text.
        
        tokens = tokenizer(
            [question] * 4, # Text A: Question (short, won't be truncated)
            [article] * 4,  # Text B: Article (long, will be truncated/strided)
            max_length=512,
            truncation="only_second", # Only truncate the article
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            padding="max_length", # Pad to max length to ensure consistent shapes if needed, or just True
            return_tensors="pt"
        )
        
        # 'tokens' will contain a flat list of all chunks for all 4 options.
        # We need to restructure them.
        # The tokenizer returns an 'overflow_to_sample_mapping' which tells us which original sample (0,1,2,3) a chunk belongs to.
        
        sample_map = tokens.pop("overflow_to_sample_mapping")
        
        # We need to group chunks by their "chunk index" (i.e., 1st chunk of opt A, 1st chunk of opt B, etc.)
        # However, the number of chunks might vary if options vary significantly in length (unlikely here) or if the stride hits boundaries differently.
        # For safety, let's assume we can group by the sequence of chunks generated.
        # Since we passed 4 pairs, and they share the exact same long context (article), they should produce the same number of chunks.
        
        # Let's verify number of chunks per option
        num_chunks = len(tokens['input_ids']) // 4
        
        # We will accumulate probabilities (or logits) across chunks.
        # Strategy: Max-Pooling or Averaging. 
        # If the evidence is in Chunk X, Chunk X should give a high prob for the correct answer. 
        # Other chunks might be ambiguous.
        # Let's try Averaging probabilities.
        
        final_probs = torch.zeros(4)
        
        # Reshape input_ids to [num_chunks, 4, seq_len]
        # The tokenizer outputs: [OptA_Chunk1, OptA_Chunk2, OptB_Chunk1, OptB_Chunk2, ...] (Sequential by sample)
        # Wait, HuggingFace tokenizer usually outputs [Sample0_Chunk0, Sample0_Chunk1, Sample1_Chunk0, ...]
        
        # Let's reorganize.
        # We have a dict of tensors.
        
        # Group inputs by chunk index
        # Expectation: inputs['input_ids'] has shape [total_chunks, seq_len]
        # We want to form batches of size 4 (one for each option) corresponding to the same text span.
        
        # Since all options are roughly same length and context is identical:
        # We assume each option generated 'num_chunks' chunks.
        
        # Create a list of batches, where each batch is a dictionary of tensors for the 4 options.
        
        chunk_batches = []
        for i in range(num_chunks):
            # We want the i-th chunk for Option 0, i-th for Option 1, etc.
            # Index in the flat list: 
            # Option 0's i-th chunk is at index: i
            # Option 1's i-th chunk is at index: i + num_chunks
            # ... NO. 
            # overflow_to_sample_mapping maps index -> sample_id.
            # Example: [0, 0, 1, 1, 2, 2, 3, 3] if each has 2 chunks.
            
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
                # Shape: [4, seq_len] -> Need [1, 4, seq_len] for the model
                b_input_ids = batch['input_ids'].unsqueeze(0)
                b_att_mask = batch['attention_mask'].unsqueeze(0)
                b_token_type = batch['token_type_ids'].unsqueeze(0) if 'token_type_ids' in batch else None
                
                outputs = model(input_ids=b_input_ids, attention_mask=b_att_mask, token_type_ids=b_token_type)
                logits = outputs.logits # [1, 4]
                probs = torch.softmax(logits, dim=1).squeeze(0) # [4]
                chunk_probs_list.append(probs)

        # Aggregate: Average Probabilities
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
