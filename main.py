import torch
import re
import json
from src import *
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import warnings
from utils import *

warnings.simplefilter("ignore")

def configurate_method(args):

    torch.manual_seed(args.seed)

    model = LLM(model=args.model, seed=args.seed, dtype=args.dtype, max_model_len=args.max_model_len)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.add_shot == '':
        print('ZERO-SHOT: OK')
    else:
        print(f'ZERO-SHOT-COT "{args.add_shot}" OK')
        
    generator = GeneratePaths(model=model, 
                              topk=args.topk, 
                              max_new_tokens=args.max_new_tokens, 
                              stop=args.stop_criteria,
                              prompt=args.add_shot)

    cot_decoding = CoTDecoding(pattern=args.pattern,
                               tokenizer=tokenizer,
                               prompt=args.add_shot)

    return generator, cot_decoding, tokenizer

def simple_demo(generator, cot_decoding, args):

    topk_tokens, outputs = generator.search_cots(args.demo_prompt)
    cot_paths = cot_decoding.calculate_score(args.demo_prompt, topk_tokens, outputs)

    return cot_paths

def evaluating_decoding_methods(args):

    generator, cot_decoding, tokenizer = configurate_method(args)
    datasets = load_datasets(args.dataset)
    
    for dataset in datasets:

        if args.log_outputs_path:
            log_outputs = {'cot_decoding': {}}
            log_outputs['cot_decoding'] = json.load(args.log_outputs_path + '/' + args.cot_name)
        else:

            if args.max_samples == -1:
                max_samples = datasets[dataset][args.field].num_rows
            else:
                max_samples = args.max_samples
            
            if args.init_samples == -1:
                init_samples = 0
            else:
                init_samples = args.init_samples
                

            log_outputs = extract_cot_paths_from_dataset(dataset=datasets[dataset],
                                            dataset_name=dataset,
                                            max_samples=max_samples,
                                            init_samples=init_samples,
                                            field=args.field,
                                            prompt_key='question',
                                            generator=generator,
                                            cot_decoding=cot_decoding      
            )

        if args.save_log:
            save_logs(log_outputs, dataset, args.log_path)
        
        ground_truth = [re.findall(args.pattern, string)[-1] for string in datasets[dataset][args.field]['answer']][init_samples:max_samples]

        evaluations = evaluating(log_outputs=log_outputs,
                                 ground_truth=ground_truth,
                                 pattern=args.pattern,
                                 num_paths=args.topk, 
                                 tokenizer=tokenizer,
                                 max_new_tokens=args.max_new_tokens)
        
        if args.save_plot:
            save_plot(evaluations, f'Exatch Match Evaluation - {dataset}', dataset)
        
        print_evaluations(evaluations, dataset)
   
if __name__ == "__main__":

    args = get_args()

    if args.choice == 'demo':
        generator, cot_decoding, tokenizer = configurate_method(args)
        cot_paths = simple_demo(generator, cot_decoding, args)
        print_output(args.demo_prompt, cot_paths)
    elif args.choice == 'evaluating':
        evaluating_decoding_methods(args)
