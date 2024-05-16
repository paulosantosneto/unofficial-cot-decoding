import os
import argparse
from datasets import load_dataset, DatasetDict
from typing import List, Dict
from tqdm import tqdm

def get_args():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default='123', type=int)
    parser.add_argument('--max_new_tokens', default=300, type=int)
    parser.add_argument('--model', default='microsoft/phi-2', type=str)
    parser.add_argument('--parallel_size', default=1, type=int)
    parser.add_argument('--remove_long_answers', default=True, type=bool)
    parser.add_argument('--topk', default=10, type=int)
    parser.add_argument('--datasets', nargs='+', choices=['gsm8k', 'multiarith', 'svamp'], default=['multiarith'], type=str)
    parser.add_argument('--split', nargs='+', choices=['train,' 'test'], type=str)
    parser.add_argument('--methods', nargs='+', choices=['gd', 'gdp', 'cd', 'cdp', 'cds', 'cdps'], default=['cds'], type=str)
    parser.add_argument('--eval', default='eval_config.yaml', type=str)
    parser.add_argument('--export_output_path', default='outputs/', type=str)
    parser.add_argument('--single_prompt', default="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?")
    parser.add_argument('--pattern', default=r'-?\b\d+(?:[,.]\d{1,10})*\b')
    parser.add_argument('--stop_criteria', default=['Q:', '\n\nQ:'])
    
    args = parser.parse_args()
    
    return args


def load_datasets(datasets: List[str]):

    loaded_datasets = {}

    # math tasks
    for dataset_name in datasets:
        if dataset_name == 'gsm8k':
            loaded_datasets['gsm8k'] = load_dataset('gsm8k', 'main')
        elif dataset_name == 'multiarith':
            loaded_datasets['multiarith'] = load_dataset('ChilleD/MultiArith')
        elif dataset_name == 'svamp':
            loaded_datasets['svamp'] = load_dataset('ChilleD/SVAMP')
    
    return loaded_datasets


def build_dataset_structure(methods):
    
    datasets = {}
    
    for method in methods:
        datasets[method] = {'question': [],
                            'reasoning': [],
                            'answer_span': [],
                            'score': []}
    return datasets

def extract_cot_decoding_from_dataset(dataset: DatasetDict,
                                  dataset_name: str,
                                  max_samples: int,
                                  question_key: str,
                                  export_to_json: bool,
                                  cot_decoding: any,
                                  step: int=100,
                                  sample: str='test'):

    dataset_sample = dataset[sample]
    
    datasets = build_dataset_structure(['gd', 'gdp', 'cd', 'cdp', 'cds', 'cdps'])

    questions = dataset_sample[question_key][:max_samples]
    
    for i, question in enumerate(tqdm(questions, desc=dataset_name, total=len(questions))):

        paths = cot_decoding.generate_text(question)
        
        cot_output = cot_decoding.extract_methods(paths)

        for key, content in cot_output.items():

            datasets[key]['question'].append(question)
            
            if len(content):
                datasets[key]['reasoning'].append(content[0]['reasoning'])
                datasets[key]['score'].append(content[0]['score'])
                datasets[key]['answer_span'].append(content[0]['answer_span'])
            else:
                datasets[key]['reasoning'].append(cot_output['gd'][0]['reasoning'])
                datasets[key]['score'].append(cot_output['gd'][0]['score'])
                datasets[key]['answer_span'].append(cot_output['gd'][0]['answer_span'])
    
    
    return datasets

def exact_match_precision(dataset_answer_span, answer_gt):
    
    return sum([answer_cot == answer for answer_cot, answer in zip(dataset_answer_span, answer_gt)]) / len(answer_gt)
    