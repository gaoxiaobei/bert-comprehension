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

    print("\nStarting comprehension task (Context Window: 1024)...\n")

    correct_count = 0
    total_questions = len(df)

    for index, row in df.iterrows():
        question = row['question']
        options = [row['choice A'], row['choice B'], row['choice C'], row['choice D']]
        correct_label = row['correct answer'].strip()
        
        print(f"Question {index + 1}: {question}")
        
        # 构造输入对
        # 通常做法是: Sentence 1 = 文章, Sentence 2 = 问题 + 选项
        # 这样模型可以基于文章来判断 问题+选项 的合理性
        first_sentences = [article] * 4
        second_sentences = [f"{question} {option}" for option in options]
        
        # Tokenize
        # max_length=1024
        # truncation="only_first": 只需要截断文章(article)，保留问题和选项完整
        inputs = tokenizer(
            first_sentences,
            second_sentences,
            max_length=1024,
            truncation="only_first", 
            padding="max_length",
            return_tensors="pt"
        )

        # 调整维度以适应模型 [Batch_Size, Num_Choices, Seq_Len]
        # 当前 inputs 的维度是 [4, 1024]，我们需要变成 [1, 4, 1024]
        model_inputs = {k: v.unsqueeze(0).to(device) for k, v in inputs.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze(0)
            
        final_probs = probs.tolist()
        predicted_class_id = probs.argmax().item()
        predicted_label = reverse_option_map[predicted_class_id]
        
        print(f"  Probabilities:")
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