
import re
import torch
import numpy as np

from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from scipy.stats import entropy
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

@dataclass
class Path:
    reasoning_text: str
    score: float
    answer_span: str
    num_path: int

@dataclass
class DecodingInfo:
    question: str
    paths: List[Path]

class GeneratePaths():

    def __init__(self, model: LLM, 
                       max_new_tokens: int=300, 
                       topk: int=10, 
                       stop: List[str]=['Q:', '\n\nQ:', '\n\nExercise'],
                       prompt: str=''):
        '''
        Args:
            - model (vllm.LLM): instance of the model in Huggingface format.
            - max_new_tokens (int): maximum number of tokens for generating the response.
            - stop (List): stopping criteria.
            - topk (int): maximum number of paths to explore for the first token of the response.
        '''
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.stop = stop
        self.topk = topk
        self.model.llm_engine.model_config.max_logprobs = self.topk + 1
        self.tokenizer = self.model.llm_engine.tokenizer.tokenizer
        self.prompt = prompt

    def search_cots(self, raw_prompt: str) -> List[str]:

        # Format the raw prompt into a predefined template format.
        formatted_prompt = self.format_prompt(raw_prompt)
        # Explore the first K paths of the response using greedy decoding.
        topk_tokens = self.get_first_topk_tokens(formatted_prompt)
        prompts = [formatted_prompt + token for token in topk_tokens['decoded']] # K questions.
        # Continue generating the K paths for the remaining M - 1 tokens.
        outputs = self.generate_paths(prompts)
        
        return topk_tokens, outputs
    
    @torch.inference_mode()
    def get_first_topk_tokens(self, prompt: str) -> Dict[str, List]:
        
        # Greedy decoding for only the first token of the response.
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
    
    @torch.inference_mode()
    def generate_paths(self, prompts: List[str]) -> Dict[int, Dict]:

        # Sampling parameters for generating paths with the model.
        sampling_params = SamplingParams(n=1, temperature=0, top_p=1, max_tokens=self.max_new_tokens, logprobs=2, stop=self.stop)
        outputs = self.model.generate(prompts, sampling_params, use_tqdm=False)

        return outputs
    
    def format_prompt(self, raw_prompt: str) -> str:
        # Format the prompt to match the expected question-answer template.
        return f'Q:{raw_prompt}\nA:{self.prompt}'

class CoTDecoding():
    
    def __init__(self, pattern: str=r'-?\d+\.?\d*',
                       tokenizer: AutoTokenizer=None,
                       prompt: str='',
                       dataset_type: str = 'aritmetic'):
        '''
        Initialize the CoTDecoding class with the given parameters.

        Args:
            answer_span_model (LLM): Model for extracting answer spans.
            pattern (str): Regex pattern for extracting numerical answer spans.
            tokenizer (AutoTokenizer): Tokenizer for encoding and decoding text.
        '''
        self.pattern = pattern
        self.tokenizer = tokenizer
        self.dataset_type = dataset_type

    def calculate_score(self, prompt, topk_tokens, outputs):
        '''
        Calculate the score for each path based on answer span probabilities.

        Args:
            prompt (str): The input prompt.
            topk_tokens (Dict): Dictionary containing the top-k tokens and their log probabilities.
            outputs (Dict): Dictionary containing the outputs from the model.

        Returns:
            DecodingInfo: An object containing the question and the scored paths.
        '''
        paths = []
        
        for k, output in enumerate(outputs):
            # Concatenate the top-k token and the generated text.
            reasoning = topk_tokens['decoded'][k] + output.outputs[0].text

            # Encode the reasoning text and obtain offset mappings.
            encode = self.tokenizer(reasoning, return_offsets_mapping=True)

            if self.dataset_type == 'aritmetic':
                answer_span = re.findall(self.pattern, reasoning)
            elif self.dataset_type == 'symbolic':
                answer_span = ''
            elif self.dataset_type == 'commonsense':
                answer_span = ''
            
            score = 0
            
            if len(answer_span):
                # Use the last found answer span.
                answer_span = answer_span[-1]
                
                # Find the position of the last pattern span in the reasoning text.
                last_pattern_span = (reasoning.rfind(answer_span), reasoning.rfind(answer_span) + len(answer_span))
                
                # Find the indices of tokens that match the answer span.
                idx_answer = [i for i, span in enumerate(encode.offset_mapping)
                              if (span[0] >= last_pattern_span[0] and span[1] <= last_pattern_span[1]) or
                              (span[0] <= last_pattern_span[0] and span[1] >= last_pattern_span[1]) or
                              (span[0] <= last_pattern_span[0] and span[1] > last_pattern_span[0])]

                token_id = [encode.input_ids[idx] for idx in idx_answer]

                output.outputs[0].logprobs.insert(0, topk_tokens['logprobs'][k])
                
                # Filter log probabilities for tokens in the answer span.
                filtered_answer = [output for i, output in enumerate(output.outputs[0].logprobs) if i in idx_answer]

                sum_answer_span_probs = 0

                for logprob_dict in filtered_answer:
                    logprob_list = list(logprob_dict.items())
                    
                    if len(logprob_list) == 2:
                        prob_diff = (torch.exp(torch.tensor([logprob_list[0][1].logprob])) - torch.exp(torch.tensor([logprob_list[1][1].logprob]))).item()
                    else:
                        prob_diff = torch.exp(torch.tensor([logprob_list[0][1].logprob])).item()
                    sum_answer_span_probs += prob_diff

                # Calculate the score as the average probability difference.
                score = 0 if len(filtered_answer) == 0 else sum_answer_span_probs / len(filtered_answer)
                answer_span = self.tokenizer.decode(token_id).strip()
            else:
                answer_span = '|<NotFounded>|'
                
            paths.append(Path(reasoning_text=reasoning, 
                              score=score,
                              answer_span=answer_span,
                              num_path=k
            ))
        
        # Create the output object with the prompt and paths.
        output = DecodingInfo(
            question=prompt,
            paths=paths
        )
        
        return output
