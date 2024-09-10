**This is an unofficial implementation of CoT-Decoding for educational and experimental purposes**. Please check out the original article ['Chain-of-Thought Reasoning Without Prompting'](https://arxiv.org/abs/2402.10200).

## Installation

```
conda create --name unofficial_cot_decoding python=3.10
conda activate unofficial_cot_decoding
pip install -r requirements.txt
```

## Experiments on Phi-2 Model

| Method | MultiArith | GSM8K | SVAMP | 
|----------|----------|----------| ----------|
| Greedy Decode | 61.1 | 32.7 | 51.0 |
| Greedy Decode + Prompt | 72.2 |  40.8 |  47.3 |
| CoT-Decoding | 75.5 | 39.8 | 41.7 |
| CoT-Decoding + Prompt | 78.3 | 46.5 | 48.0 |
| CoT-Decoding + Self Consistency | **96.7** |  **54.7** |  **67.0** |
| CoT-Decoding + Self Consistency + Prompt | 94.4 | **54.7** | 66.7 |

```
python3 main.py --max_new_tokens 300 \
                --model microsoft/phi-2 \
                --topk 10 \
                --datasets multiarith svamp \
                --methods gd cd cds
```


**Arguments**

- `--max_new_tokens`: maximum number of tokens generated in the autoregression process.
- `--topk`: number of top greedy decode paths to explore.
- `--model`: name of the model to be loaded during execution (Support only for HF models).
- `--methods`: allows execution in two subcategories: standard and prompt. Standard refers to the set of greedy decode, cot-decoding, and cot-decoding with self-consistency. Prompt addresses the same methods with the addition of the specific prompt (e.g., "Let's think step by step").
- `--seed`: for reproducibility purposes.
- `--parallel_size`: number of GPUs for parallelization.
- `--remove_long_reasoning`: boolean term for removing responses that reach the maximum number of new tokens.
