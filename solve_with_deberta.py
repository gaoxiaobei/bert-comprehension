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
    parser.add_argument("--max_len", type=int, default=1024, help="Manual maximum context window limit.")
    
    args = parser.parse_args()

    # File paths
    article_path = args.article
    qa_path = args.qa
    model_name = args.model
    user_max_limit = args.max_len  # 用户自定义的上限，忽略模型config中的512

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    reverse_option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    print(f"Configuration: Using dynamic window with a hard cap of {user_max_limit} tokens.")
    print("\nStarting comprehension task...\n")

    correct_count = 0
    total_questions = len(df)

    for index, row in df.iterrows():
        question = row['question']
        options = [row['choice A'], row['choice B'], row['choice C'], row['choice D']]
        correct_label = row['correct answer'].strip()
        
        print(f"=== Question {index + 1} ===")
        
        # 1. 预计算当前题目需要的 Token 长度
        # 不设限，测量实际长度
        temp_inputs = tokenizer(
            [article] * 4,
            [f"{question} {opt}" for opt in options],
            truncation=False
        )
        
        actual_max_needed = max([len(x) for x in temp_inputs['input_ids']])
        
        # 2. 自动适配 Context Window
        # 取 (实际需要) 和 (用户指定的 1024) 的最小值
        dynamic_window = min(actual_max_needed, user_max_limit)
        
        print(f"  [Token Stats] Actual needed: {actual_max_needed} tokens.")
        print(f"  [Window Size] Adapted to: {dynamic_window} tokens.")

        # 3. 正式 Tokenize
        inputs = tokenizer(
            [article] * 4,
            [f"{question} {opt}" for opt in options],
            max_length=dynamic_window,
            truncation="only_first", # 重点：若超长，只砍掉文章，保留问题和选项
            padding="max_length",
            return_tensors="pt"
        )

        # --- Input Preview ---
        # 预览第一个选项的解码效果，确保 [SEP] 后面能看到问题
        input_ids_opt_a = inputs['input_ids'][0]
        decoded_text = tokenizer.decode(input_ids_opt_a)
        
        # 截取前后预览
        preview_len = 120
        if len(decoded_text) > preview_len * 2:
            preview_str = f"{decoded_text[:preview_len]} ... [HIDDEN] ... {decoded_text[-preview_len:]}"
        else:
            preview_str = decoded_text
        print(f"  [Input Preview] {preview_str}")

        # 4. 推理
        model_inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            # 注意：DeBERTa 处理超过 512 的序列时，显存占用会激增
            outputs = model(**model_inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        final_probs = probs.tolist()
        predicted_class_id = probs.argmax().item()
        predicted_label = reverse_option_map[predicted_class_id]
        
        print(f"  [Results]")
        for i, prob in enumerate(final_probs):
            print(f"    {reverse_option_map[i]}: {prob:.4f}")
        
        print(f"  Predicted: {predicted_label} | Correct: {correct_label}")
        
        if predicted_label == correct_label:
            print("  Result:    CORRECT")
            correct_count += 1
        else:
            print("  Result:    INCORRECT")
        print("-" * 60)

    print(f"\nFinished. Accuracy: {correct_count}/{total_questions} ({correct_count/total_questions*100:.2f}%)")

if __name__ == "__main__":
    main()