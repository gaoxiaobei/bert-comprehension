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
    parser.add_argument("--fixed", action="store_true", help="If set, force the window to always be max_len.")
    
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

    # 读取文件
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
        choices = {
            'A': row['choice A'],
            'B': row['choice B'],
            'C': row['choice C'],
            'D': row['choice D']
        }
        options = [choices['A'], choices['B'], choices['C'], choices['D']]
        correct_label = row['correct answer'].strip()
        
        print(f"=== Question {index + 1} ===")
        # 打印原始题目信息
        print(f"  [Q]: {question}")
        for label, text in choices.items():
            print(f"  [{label}]: {text}")
        
        # 1. 预计算实际需要的 Token 长度
        temp_inputs = tokenizer(
            [article] * 4,
            [f"{question} {opt}" for opt in options],
            truncation=False
        )
        actual_needed = max([len(x) for x in temp_inputs['input_ids']])
        
        # 2. 确定最终使用的窗口大小
        final_window = user_max_limit if args.fixed else min(actual_needed, user_max_limit)
        
        # 3. 正式编码
        inputs = tokenizer(
            [article] * 4,
            [f"{question} {opt}" for opt in options],
            max_length=final_window,
            truncation="only_first", 
            padding="max_length",
            return_tensors="pt"
        )

        # 4. Token 统计与改进后的预览 (跳过 PAD)
        valid_tokens = inputs['attention_mask'][0].sum().item()
        # 仅针对非 PAD 的内容进行解码预览
        valid_ids = inputs['input_ids'][0][inputs['attention_mask'][0] == 1]
        decoded_content = tokenizer.decode(valid_ids, skip_special_tokens=False)
        
        print(f"  --- Inference Info ---")
        print(f"  Window: {final_window} | Valid Tokens: {valid_tokens} | Needed: {actual_needed}")
        
        # 预览内容末尾 (即问题+选项拼接处)
        preview_tail = decoded_content[-150:].replace('\n', ' ')
        print(f"  [Tail Preview]: ... {preview_tail}")

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
        
        # 6. 结果输出
        print(f"  [Model Probabilities]")
        for i, prob in enumerate(final_probs):
            tag = " <--" if i == predicted_class_id else ""
            print(f"    {reverse_option_map[i]}: {prob:.4f}{tag}")
        
        print(f"  Result: Predicted [{predicted_label}] | Correct [{correct_label}]", end=" ")
        
        if predicted_label == correct_label:
            print("CORRECT ✅")
            correct_count += 1
        else:
            print("INCORRECT ❌")
        print("-" * 70)

    accuracy = (correct_count / total_questions) * 100
    print(f"\nTask Finished.")
    print(f"Final Accuracy: {correct_count}/{total_questions} ({accuracy:.2f}%)")

if __name__ == "__main__":
    main()