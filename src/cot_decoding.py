import re
import torch

from typing import List, Dict
from vllm import LLM, SamplingParams

class COTDecoding():

    def __init__(self, model, max_new_tokens: int, pattern: str, topk: int, stop: List[str], methods: List[str], template: str='standard'):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.stop = stop
        self.pattern = pattern
        self.format_question = self.standard_template
        self.topk = topk
        self.verbose = True
        self.methods = methods
        self.model.llm_engine.model_config.max_logprobs = self.topk + 1
        self.tokenizer = self.model.llm_engine.tokenizer.tokenizer

        if template == 'prompt': self.format_question = self.prompt_template
    
    def extract_methods(self, paths: Dict[str, str]):
        methods = {}
        # greedy decode
        if 'gd' in self.methods:
            methods['gd'] = [path for path in paths if path['k'] == 0]
        # cot-decoding
        if 'cd' in self.methods:
            methods['cd'] = [max(paths[:limit], key=lambda x: x['score'])  for limit in range(1, len(paths)+1)]
        # cot-decoding + self consistency 
        if 'cds' in self.methods:

            consistency = {}

            for path in paths:
                if path['answer_span'] not in consistency:
                    consistency[path['answer_span']] = 0
                consistency[path['answer_span']] += path['score']

            major_answer_span = max(consistency, key=consistency.get)
            methods['cds'] = [max([item for item in paths if item['answer_span'] == major_answer_span], key=lambda x: x['score'])]

        return methods
        
    def generate_text(self, prompt: str):

        prompt = self.format_question(prompt)

        topk_tokens = self.get_first_topk_tokens(prompt)

        prompts = [prompt + token for token in topk_tokens['decoded']]

        outputs = self.generate_paths(prompts)

        paths = self.get_paths(topk_tokens, outputs)

        return paths

    def get_paths(self, topk_tokens: Dict[str, any], outputs: List) -> List[Dict[str, any]]:

        paths = []

        for k, output in enumerate(outputs):

            reasoning = topk_tokens['decoded'][k] + output.outputs[0].text
            encode = self.tokenizer(reasoning, return_offsets_mapping=True)
            pattern_found = re.findall(self.pattern, reasoning)

            if len(pattern_found):
            
                last_pattern_span = (reasoning.rfind(pattern_found[-1]), reasoning.rfind(pattern_found[-1]) + len(pattern_found[-1]))

                idx_answer = [i for i, span in enumerate(encode.offset_mapping) 
                            if (span[0] >= last_pattern_span[0] and span[1] <= last_pattern_span[1]) or
                            (span[0] <= last_pattern_span[0] and span[1] >= last_pattern_span[1]) or
                            (span[0] <= last_pattern_span[0] and span[1] > last_pattern_span[0])]

                token_id = [encode.input_ids[idx] for idx in idx_answer]

                output.outputs[0].logprobs.insert(0, topk_tokens['logprobs'][k])
                
                filtered_answer = [output for i, output in enumerate(output.outputs[0].logprobs) if i in idx_answer]

                sum_answer_span_probs = 0

                for logprob_dict in filtered_answer:
                    logprob_list = list(logprob_dict.items())
                    if len(logprob_list) == 2:
                        prob_diff = (torch.exp(torch.tensor([logprob_list[0][1].logprob])) - torch.exp(torch.tensor([logprob_list[1][1].logprob]))).item()
                    else:
                        prob_diff = torch.exp(torch.tensor([logprob_list[0][1].logprob])).item()
                    sum_answer_span_probs += prob_diff
                
                #score = 0 if len(filtered_answer) == 0 else sum_answer_span_probs / len(filtered_answer)
                score = sum_answer_span_probs / len(filtered_answer)
                answer_span = self.tokenizer.decode(token_id).strip()
            else:
                score = 0
                answer_span = self.tokenizer.eos_token

            paths.append({'score': score,
                          'reasoning': reasoning,
                          'answer_span': answer_span,
                          'k': k})

        return paths
        
    @torch.inference_mode()
    def generate_paths(self, prompts: List[str]):

        sampling_params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
        outputs = self.model.generate(prompts, sampling_params, use_tqdm=False)

        return outputs

    @torch.inference_mode()
    def get_first_topk_tokens(self, prompt: str):

        sampling_params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=1, logprobs=self.topk, stop=self.stop)
        outputs = self.model.generate(prompt, sampling_params, use_tqdm=False)[0].outputs[0].logprobs[0]

        topk_tokens = {'decoded': [], 'probs': [], 'token_id': [], 'logprobs': []}

        for token_id, logprob_obj in outputs.items():
            
            topk_tokens['logprobs'].append({token_id: logprob_obj})
            topk_tokens['decoded'].append(logprob_obj.decoded_token)
            topk_tokens['probs'].append(logprob_obj.logprob)
            topk_tokens['token_id'].append(token_id)

        topk_tokens['probs'] = torch.exp(torch.tensor(topk_tokens['probs'])).tolist()

        return topk_tokens

    def standard_template(self, prompt: str):
        return f"Q:{prompt}\nA:"

    def prompt_template(self, prompt: str):
        return f"Q:{prompt}\nA: Let's think step by step."
