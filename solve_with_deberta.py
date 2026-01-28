import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMultipleChoice
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Solve reading comprehension using DeBERTa.")
    parser.add_argument("--article", default="article.md", help="Path to the article file.")
    parser.add_argument("--qa", default="QA.csv", help="Path to the QA CSV file.")
    parser.add_argument("--model", default="artianand/deberta-v3-large-race", help="Hugging Face model name.")
    parser.add_argument("--max_len", type=int, default=1024, help="Maximum allowed context window.")
    parser.add_argument("--fixed", action="store_true", help="If set, force the window to always be max_len. Otherwise, use dynamic window.")
    
    args = parser.parse_args()

    # 配置
    article_path = args.article
    qa_path = args.qa
    model_name = args.model
    user_max_limit = args.max_len 

    if not os.path.exists(article_path):
        print(f"Error: Article file '{article_path}' not found.")
        return
    if not os.path.exists(qa_path):
        print(f"Error: QA file '{qa_path}' not found.")
        return

    # 读取内容
    with open(article_path, "r", encoding="utf-8") as f:
        article = f.read().strip()
    df = pd.read_csv(qa_path)

    # 加载模型
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

    print(f"Mode: {'FIXED' if args.fixed else 'DYNAMIC'} Window")
    print(f"Max Limit: {user_max_limit} tokens")
    print(f"Device: {device}\n")

    correct_count = 0
    total_questions = len(df)

    for index, row in df.iterrows():
        question = row['question']
        options = [row['choice A'], row['choice B'], row['choice C'], row['choice D']]
        correct_label = row['correct answer'].strip()
        
        print(f"=== Question {index + 1} ===")
        
        # 1. 预计算实际需要的 Token 长度
        temp_inputs = tokenizer(
            [article] * 4,
            [f"{question} {opt}" for opt in options],
            truncation=False
        )
        actual_needed = max([len(x) for x in temp_inputs['input_ids']])
        
        # 2. 确定最终使用的窗口大小
        if args.fixed:
            final_window = user_max_limit
        else:
            final_window = min(actual_needed, user_max_limit)
        
        print(f"  [Window Config] Actual needed: {actual_needed} | Selected: {final_window}")

        # 3. 正式编码
        # truncation="only_first" 确保文章太长时从头部截断，保留尾部的问题和选项
        inputs = tokenizer(
            [article] * 4,
            [f"{question} {opt}" for opt in options],
            max_length=final_window,
            truncation="only_first", 
            padding="max_length",
            return_tensors="pt"
        )

        # 4. 输入预览 (查看末尾是否包含问题和选项)
        # 预览第一个选项 (A)
        valid_ids = inputs['input_ids'][0][inputs['attention_mask'][0] == 1]
        full_decoded_valid = tokenizer.decode(valid_ids, skip_special_tokens=False)
        # 统计实际有效的非 padding token
        valid_tokens = inputs['attention_mask'][0].sum().item()
        
        print(f"  [Token Stats] Valid non-padding tokens: {valid_tokens}")
        # 截取最后120个字符作为预览
        preview_tail = full_decoded_valid[-150:].replace('\n', ' ')
        print(f"  [Tail Preview] ... {preview_tail}")

        # 5. 模型推理
        model_inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        final_probs = probs.tolist()
        predicted_class_id = probs.argmax().item()
        predicted_label = reverse_option_map[predicted_class_id]
        
        # 6. 结果展示
        print(f"  [Probabilities]")
        for i, prob in enumerate(final_probs):
            tag = " <- PREDICTED" if i == predicted_class_id else ""
            print(f"    {reverse_option_map[i]}: {prob:.4f}{tag}")
        
        print(f"  Result: {predicted_label} (Correct: {correct_label})", end=" ")
        
        if predicted_label == correct_label:
            print("[CORRECT ✅]")
            correct_count += 1
        else:
            print("[INCORRECT ❌]")
        print("-" * 65)

    accuracy = (correct_count / total_questions) * 100
    print(f"\nFinal Accuracy: {correct_count}/{total_questions} ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()