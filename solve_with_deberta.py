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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMultipleChoice.from_pretrained(model_name)
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Map for options
    reverse_option_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    max_seq_length = 1024

    print(f"\nStarting comprehension task (Context Window: {max_seq_length})...\n")

    correct_count = 0
    total_questions = len(df)

    for index, row in df.iterrows():
        question = row['question']
        options = [row['choice A'], row['choice B'], row['choice C'], row['choice D']]
        correct_label = row['correct answer'].strip()
        
        print(f"=== Question {index + 1} ===")
        print(f"Q: {question}")
        
        # 构造输入对
        first_sentences = [article] * 4
        second_sentences = [f"{question} {option}" for option in options]
        
        # Tokenize
        inputs = tokenizer(
            first_sentences,
            second_sentences,
            max_length=max_seq_length,
            truncation="only_first", # 保证只截断文章，保留问题和选项
            padding="max_length",
            return_tensors="pt"
        )

        # --- Debug: Token Stats & Preview ---
        # 计算实际 Token 长度 (通过 attention_mask 求和)
        real_token_counts = inputs['attention_mask'].sum(dim=1).tolist()
        
        print(f"  [Token Stats]")
        for i, count in enumerate(real_token_counts):
            status = "TRUNCATED" if count == max_seq_length else "OK"
            print(f"    Option {reverse_option_map[i]}: {count} tokens ({status})")
        
        # 预览第一个选项的输入内容 (解码)
        # 这里的目的是检查：文章开头是否还在？最重要的问题和选项是否在末尾？
        print(f"  [Input Preview - Option A]")
        input_ids_opt_a = inputs['input_ids'][0]
        # 解码所有非 padding 的部分
        decoded_text = tokenizer.decode(input_ids_opt_a[inputs['attention_mask'][0] == 1])
        
        # 为了不刷屏，只显示 开头 150字符 ... 结尾 150字符
        preview_len = 150
        if len(decoded_text) > preview_len * 2:
            preview_str = f"{decoded_text[:preview_len]} ... [CONTENT HIDDEN] ... {decoded_text[-preview_len:]}"
        else:
            preview_str = decoded_text
            
        print(f"    Raw Input: \"{preview_str}\"\n")
        # ------------------------------------

        # 调整维度 [Batch, Choices, Seq]
        model_inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        final_probs = probs.tolist()
        predicted_class_id = probs.argmax().item()
        predicted_label = reverse_option_map[predicted_class_id]
        
        print(f"  [Results]")
        for i, prob in enumerate(final_probs):
            print(f"    {reverse_option_map[i]}: {prob:.4f}")
        
        print(f"  Predicted: {predicted_label}")
        print(f"  Correct:   {correct_label}")
        
        if predicted_label == correct_label:
            print("  Result:    CORRECT")
            correct_count += 1
        else:
            print("  Result:    INCORRECT")
        print("-" * 50)

    print(f"\nFinished. Accuracy: {correct_count}/{total_questions} ({correct_count/total_questions*100:.2f}%)")

if __name__ == "__main__":
    main()