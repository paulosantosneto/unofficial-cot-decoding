import torch

from src import *
from vllm import LLM, SamplingParams
import warnings

# Suppress all warnings globally
warnings.simplefilter("ignore")

def run_cot_decoding(args):

    torch.manual_seed(args.seed)

    model = LLM(model=args.model, seed=args.seed)
    

    cot_decoding = COTDecoding(model=model, 
                               max_new_tokens=300, 
                               pattern=args.pattern, 
                               topk=args.topk,
                               stop=args.stop_criteria,
                               methods=args.methods)

    loaded_datasets = load_datasets(args.datasets)

    dataset_output = extract_cot_decoding_from_dataset(dataset=loaded_datasets['multiarith'],
                                      dataset_name='multiarith',
                                      max_samples=loaded_datasets['multiarith']['test'].num_rows,
                                      sample='test',
                                      question_key='question',
                                      export_to_json=False,
                                      cot_decoding=cot_decoding)
    
    precision = exact_match_precision(dataset_output['cds']['answer_span'], loaded_datasets['multiarith']['test']['final_ans'])

    print(f'Precision: {precision:.3f}')

if __name__ == "__main__":

    args = get_args()

    run_cot_decoding(args)
