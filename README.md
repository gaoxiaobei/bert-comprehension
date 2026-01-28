# BERT-Comprehension

Use BERT models to solve comprehension problems.

## Usage
### Prepare Requirements

`pip install -r requirements.txt`

### Run Inference

- `python solve_with_deberta.py` (Recommended)
    - This will use `artianand/deberta-v3-large-race` to solve problems. It uses relative context position, so dynamic window is turned on by default, adapting to the actual length of problems.

- `python solve_comprehension.py` (Less competitive)
    - This will use `Riiid/kda-albert-xxlarge-v2-race` to solve problems. It supports 512-tokens context window. Uses slide window to solve longer problems.

### Parameters

```bash
usage: solve_comprehension.py [-h] [--article ARTICLE] [--qa QA] [--model MODEL] [--batch_size BATCH_SIZE]

Solve reading comprehension with dynamic aggregation strategies (Optimized).

options:
  -h, --help            show this help message and exit
  --article ARTICLE     Path to the article file.
  --qa QA               Path to the QA CSV file.
  --model MODEL         Hugging Face model name.
  --batch_size BATCH_SIZE
                        Inference batch size (number of windows processed at once).
```
```bash
usage: solve_with_deberta.py [-h] [--article ARTICLE] [--qa QA] [--model MODEL] [--max_len MAX_LEN] [--fixed]

Solve reading comprehension using DeBERTa.

options:
  -h, --help         show this help message and exit
  --article ARTICLE  Path to the article file.
  --qa QA            Path to the QA CSV file.
  --model MODEL      Hugging Face model name.
  --max_len MAX_LEN  Maximum allowed context window.
  --fixed            If set, force the window to always be max_len.
```