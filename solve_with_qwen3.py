import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", default="cet-6-202506.md")
    parser.add_argument("--qa", default="cet6-2506.csv")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen3 (Thinking Mode Enabled): {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype="auto", 
        device_map="auto"
    )

    with open(args.article, "r", encoding="utf-8") as f:
        article = f.read().strip()
    df = pd.read_csv(args.qa)

    correct_count = 0
    
    for idx, row in df.iterrows():
        question = row['question']
        choices = f"A) {row['choice A']}\nB) {row['choice B']}\nC) {row['choice C']}\nD) {row['choice D']}"
        correct_answer = row['correct answer'].strip()

        # Instruction to force standardization
        prompt = f"Article:\n{article}\n\nQuestion: {question}\nChoices:\n{choices}\n\nReason step by step and provide the final choice letter in the format 'Answer: [Letter]'."

        messages = [{"role": "user", "content": prompt}]
        
        # enable_thinking=True is the key feature of Qwen3
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True 
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(device)

        with torch.no_grad():
            # Reasoning models need room to "think", so we set a higher max_new_tokens
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=2048,
                temperature=0.6, # Best practice for thinking mode
                top_p=0.95,
                top_k=20
            )
            
            # Remove the input tokens from the output
            output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()

        # Qwen3 uses special tokens for thinking. 
        # 151667 is <think>, 151668 is </think>
        try:
            think_end_idx = output_ids.index(151668)
            thinking_content = tokenizer.decode(output_ids[:think_end_idx], skip_special_tokens=True).strip()
            final_content = tokenizer.decode(output_ids[think_end_idx+1:], skip_special_tokens=True).strip()
        except ValueError:
            thinking_content = "No thinking block found."
            final_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Extract [A-D] from the final response
        match = re.search(r'Answer:\s*\[?([A-D])\]?', final_content, re.IGNORECASE)
        predicted_label = match.group(1).upper() if match else "N/A"

        print(f"\n=== Q{idx+1}: {question} ===")
        print(f"  [Thinking]: {thinking_content[:200]}...") # Print first 200 chars of reasoning
        print(f"  [Model Choice]: {predicted_label} | [Correct]: {correct_answer}", end=" ")
        
        if predicted_label == correct_answer:
            print("✅")
            correct_count += 1
        else:
            print(f"❌ (Model said: {final_content[:50]}...)")

    accuracy = (correct_count / len(df)) * 100
    print(f"\nFinal Accuracy: {correct_count}/{len(df)} ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()