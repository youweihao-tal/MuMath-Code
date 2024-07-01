"""
This script support vllm batch inference with cot/pal/tora/mumathcode prompt.
Code based on: https://github.com/microsoft/ProphetNet/tree/master/CRITIC and https://github.com/microsoft/tora. 
"""
import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

from eval.evaluate import evaluate
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data 
from utils.python_executor import PythonExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="gsm8k", type=str)
    parser.add_argument("--data_dir", default="./data", type=str) 
    parser.add_argument("--data_file", type=str) 
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str) 
    parser.add_argument("--output_dir", type=str) 
    parser.add_argument("--output_file", type=str) 
    parser.add_argument("--prompt_type", default="mumathcode", type=str) 
    parser.add_argument("--split", default="test", type=str) 
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data 
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_train_prompt_format", action="store_true")
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    return args


def main(args):
    examples = load_data(args.data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)
    elif args.num_test_sample == -1:
        args.num_test_sample = len(examples)
    
    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    if args.end == -1:
        args.end = len(examples)
    examples = examples[args.start:args.end]

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    # load model
    if len(examples) > 0:
        available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        llm = LLM(model=args.model_name_or_path, tensor_parallel_size=len(available_gpus)) 
    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, args.data_name)
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name) 
        full_prompt = construct_prompt(args, example)

        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}

        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample) 

    print("dataset:", args.data_name, "samples:", len(samples))
    if len(samples) > 0:
        print("-" * 50)
        print("sample:", samples[0]['prompt'])
        print("-" * 50)

    # repeat n times
    remain_prompts = [sample['prompt'] for sample in samples for _ in range(args.n_sampling)] 
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else 4  
    stop_tokens = ["</s>", "```output"]

    if args.prompt_type in ['cot']:
        stop_tokens.append("\n\n")
    elif args.prompt_type in ['wizard_zs', 'platypus_fs']:
        stop_tokens.extend(["Instruction", "Response"]) 

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("=" * 50, "Epoch", epoch)
        current_prompts = remain_prompts 
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts] 
        outputs = llm.generate(prompts, SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens_per_call,
                        n=1, 
                        stop=stop_tokens,
        ))

        outputs = sorted(outputs, key=lambda x: int(x.request_id)) 
        outputs = [output.outputs[0].text for output in outputs]  
        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output 
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))

            elif ("boxed" not in output and output.endswith("```")): 
                program = extract_program(query)
                remain_prompts.append((i, query)) 
                remain_codes.append(program)
            else: 
                end_prompts.append((i, query)) 

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k] 
            res, report = remain_results[k]
            exec_result = res if res else report 
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result # \tau_i + r_i + o_i -> \tau_{i+1} 
            # not end
            if epoch == max_func_call - 1: 
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query) 

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0]) 
    ans_split = "<|assistant|>" if args.use_train_prompt_format else "Question:"
    codes = [prompt.split(ans_split)[-1].strip() for _, prompt in end_prompts] 

    # extract preds
    results = [run_execute(executor, code, args.prompt_type) for code in codes] 
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i*args.n_sampling: (i+1)*args.n_sampling]
        result = results[i*args.n_sampling: (i+1)*args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]

        sample.pop('prompt')
        sample.update({'code': code, 'pred': preds, 'report': reports})
        all_samples.append(sample)

    # add processed samples
    # all_samples.extend(processed_samples)  
    out_file = os.path.join(args.output_dir, args.output_file + ".jsonl") 
    save_jsonl(all_samples, out_file) 

    result_str = evaluate(samples=all_samples, data_name=args.data_name, prompt_type=args.prompt_type, execute=True)
    result_str += f"\nTime use: {time_use:.2f}s"
    time_str = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    result_str += f"\nTime use: {time_str}"

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}.metrics"), "w") as f: 
        f.write(result_str) 

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
