**This is an unofficial implementation of CoT-Decoding for educational and experimental purposes**. Please check out the original article ['Chain-of-Thought Reasoning Without Prompting'](https://arxiv.org/abs/2402.10200).

A more accessible demo is also available at [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JGY37hfAIoV0Om_frXZfQXzYoHvAmDkj?usp=sharing).


## Local

```
conda create --name unofficial_cot_decoding python=3.10
conda activate unofficial_cot_decoding
pip install -r requirements.txt
```

## Demo
```
!python3 main.py --choice demo --model microsoft/phi-2 --demo_prompt "I have 3 apples, my dad has 2 more apples than me, how many apples do we have in total?" --topk 10
```

Output:

```
Your prompt: I have 3 apples, my dad has 2 more apples than me, how many apples do we have in total?

CoT-Decoding Paths:

	(k=0) Reasoning Text:  We have 5 apples in total.
 (Score: 0.6773) (Span: 5)
	(k=1) Reasoning Text:  You have 3 apples and your dad has 5 apples, so you have 8 apples in total.
 (Score: 0.3466) (Span: 8)
	(k=2) Reasoning Text:  To find the total number of apples, we need to add the number of apples you have (3) to the number of apples your dad has (2 more than you, which is 3 + 2 = 5). So, you and your dad have a total of 3 + 5 = 8 apples.

 (Score: 0.9845) (Span: 8)
	(k=3) Reasoning Text:  Your dad has 5 apples and you have 3 apples, so you have 8 apples in total.
 (Score: 0.3692) (Span: 8)
	(k=4) Reasoning Text: We have 5 apples in total.
 (Score: 0.7439) (Span: 5)
	(k=5) Reasoning Text:  5 apples.
 (Score: 0.0358) (Span: 5)
	(k=6) Reasoning Text:  Let's use algebra to solve this problem. Let's say the number of apples you have is x. Then, your dad has x + 2 apples. The total number of apples is x + (x + 2) = 2x + 2. If you have 3 apples, then x = 3. Therefore, the total number of apples is 2(3) + 2 = 8.
 (Score: 0.9302) (Span: 8.)
	(k=7) Reasoning Text:  My dad has 5 apples and I have 3 apples, so we have 8 apples in total.

 (Score: 0.4614) (Span: 8)
	(k=8) Reasoning Text:  The total number of apples is 5.
 (Score: 0.8735) (Span: 5.)
	(k=9) Reasoning Text:  I have 3 apples and my dad has 5 apples, so we have 8 apples in total.

 (Score: 0.4824) (Span: 8)
```

## Evaluation
```
!python3 main.py --choice evaluating --model microsoft/phi-2 --topk 10 --dataset multiarith
```

Output:

```
--- Evaluation using Exact Match for multiarith ---
	Greedy Decoding: 0.6000
	CoT-Decoding (max): 0.7278
	CoT-Decoding (agg): 0.9444
```

**Arguments**

- `--max_new_tokens`: maximum number of tokens generated in the autoregression process.
- `--topk`: number of top greedy decode paths to explore.
- `--model`: name of the model to be loaded during execution (Support only for HF models).
- `--seed`: for reproducibility purposes.