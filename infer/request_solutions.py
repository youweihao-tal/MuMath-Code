

"""
Multi-threading request for code nested solutions to math questions.

This script support LLM API inference with MuMath-Code prompt.
It can be used to generate MuMath-Code-Data.
Code based on: https://github.com/microsoft/ProphetNet/tree/master/CRITIC and https://github.com/microsoft/tora. 
"""

import json
import random
import os
import pprint
import re
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from sympy.printing.pretty import pretty 

from utils.python_executor import PythonExecutor
from utils.utils import *
from utils.parser import *
from eval.grader import * 

from utils.data_loader import load_data_mumathcode 

from typing import Union, List 
import time 
import requests 
import time 
import threading 


# response_i: the i-th generated content of the LLM, including the i-th python code block, i.e., code_i 
# execution_i: the i-th execution output of the python interpreter (the tool) 


# We adopt URL to request the proprietary LLM, e.g., GPT-4. The users can rewrite the llm_api function according to their LLM requesting manners. 
url = "xxxxxxxx" 
headers = { 
	"Content-Type": "application/json",
	"Username": "xxxxxx",
	"Password": "xxxxxx"
}

correction_prompt = "The code above has encountered a problem. Now point out its mistakes and then correct them.\n" 
gpt_correction_prompt_suffix = "Your generated new python code should in the same format as before, like:\n```python\n[Your generated code]\n```\n```output\n[The executed output]\n```" # prompt LLM to generate a "```output" after every code block 

gpt_wrap_answer_prompt_suffix = "Present the final result in LaTeX using a `\\boxed{}` without any units.\n" 


def logging(logger, content): 
    if logger: 
        logger.write(content) 
        logger.flush() 

def format_time(seconds): 
    seconds = int(seconds) 
    
    minutes = seconds // 60 
    seconds = seconds % 60 

    hours = minutes // 60 
    minutes = minutes % 60 

    days = hours // 24 
    hours = hours % 24 

    return f"{days}d {hours}h {minutes}m {seconds}s" 

# We request the LLM via URL. The users can rewrite this function according to their LLM requesting manners. 
def llm_api(prompt: str, max_tokens: int, temperature: float, n: int, top_p: float, stop: Union[List[str], None], logger=None): 
    data = { 
		'messages': [ 
			{
				"role": "user", 
				"content": prompt 
			}
		], 
        'max_tokens': max_tokens, # max_tokens for every call 
        'temperature': temperature, 
        'n': n, 
        'top_p': top_p, 
        'stop': stop 
	} 
    url_response = requests.post(url, json=data, headers=headers) 
    
    if url_response.status_code == 200: 
        response_json = url_response.json() 
        choices = response_json["choices"] 
        rs = [] 
        for c in choices: 
            rs.append(c["message"]["content"]) 
        return rs 
    elif url_response.status_code == 429: 
        # print("Too many requests. Waiting and retrying...") 
        logging(logger, '.') 
        retry_after = int(url_response.headers.get("Retry-After", 5)) 
        # time.sleep(retry_after*2) 
        time.sleep(retry_after/3) 
        return llm_api(prompt, max_tokens, temperature, n, top_p, stop, logger) 
    else: 
        time.sleep(1) 
        logging(logger, 'x') 
        return llm_api(prompt, max_tokens, temperature, n, top_p, stop, logger) 


def api_with_func_call(prompt, max_tokens, temperature, n, top_p, executor, max_func_call=4, logger=None):
    if n > 1:
        assert temperature > 0 

    next_batch_queries = [""] * n 
    execution_success = [False] * n 
    end_queries = [] 
    for i in range(max_func_call): 
        batch_outputs = [] 
        batch_queries = next_batch_queries 
        if len(batch_queries) == 0: 
            break 

        pre_execution_success = execution_success 

        for k, query in enumerate(batch_queries):
            start_time = time.time() 

            suffix = "" 
            if i == 0: 
                pass # do nothing 
            elif pre_execution_success[k]:
                suffix = query + gpt_wrap_answer_prompt_suffix 
            else: # i > 0 and pre_execution_success[k] == False 
                suffix = query + gpt_correction_prompt_suffix 

            results = llm_api( 
                prompt=prompt + suffix, 
                max_tokens=max_tokens, 
                temperature=temperature,
                n=1, 
                top_p=top_p, 
                stop=["```output\n", "---"], # excluding "```", because the responses should finally contain "```" 
                logger=logger 
            ) 
            batch_outputs.append(results[0]) 
            end_time = time.time() 
            delta_time = end_time - start_time 
            logging(logger, "Time of the {}-th/{} sampling in the {}-th call for one question: ".format(k+1, len(batch_queries), i+1) + format_time(delta_time) + '\n') 

        # process all outputs
        next_batch_queries = [] 
        execution_success = [] 
        for k, t in enumerate(zip(batch_queries, batch_outputs)): 
            query, output = t 
            output = output.rstrip() 
            query += output # response_1 + execution_1 + ... + reseponse_{i-1} + execution_{i-1} + response_i 

            if "\\boxed" not in output and output.endswith("```"): # response_i does not include the final answer 
                program = extract_program(query) 

                start_time = time.time() 
                prediction, report = executor.apply(program) # execution_i 
                end_time = time.time() 
                delta_time = end_time - start_time 
                logging(logger, "Time of the {}-th/{} sampling in the {}-th call for one question: ".format(k+1, len(batch_queries), i+1) + '{:.6f} sec\n'.format(delta_time) ) 

                exec_result = prediction if prediction else report
                exec_result = f"\n```output\n{exec_result.strip()}\n```\n"
                query += exec_result # response_1 + execution_1 + ... + response_i + execution_i 

                # Judge runtime error occurs. If it happens, append the correcting prompt 
                if not prediction: # runtime error 
                    query += correction_prompt 
                    execution_success.append(False) 
                else: # executed successfully 
                    execution_success.append(True) 
                # not end 
                if i == max_func_call - 1:
                    query += "\nReach max function call limit." 
                next_batch_queries.append(query) 
            else:
                end_queries.append(query) # response_i includes the final answer 
    end_queries.extend(next_batch_queries) 
    return end_queries 


def tread_processer(args, split_jsonl_data, start_index, end_index, raw_output_dir):

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()') 
    else: 
        executor = PythonExecutor(get_answer_from_stdout=True) 

    out_file = os.path.join(raw_output_dir, "zz_{}_{}.jsonl".format(start_index, end_index) ) 
    writer = open(out_file, 'w') 
    
    logger = None 
    if args.debug: 
        log_file = os.path.join(raw_output_dir, "zz_{}_{}.log".format(start_index, end_index) ) 
        logger = open(log_file, 'w') 

    for example in tqdm(split_jsonl_data, desc="split_{}_{}".format(str(start_index), str(end_index))): 
    # for example in split_jsonl_data: 
        
        idx = example['idx'] 
        logging(logger, '\n################### problem {} starts #####################\n'.format(idx)) 
        
        # parse question and answer 
        example['question'] = parse_question_mumathcode(example, args.data_name) 
        
        full_prompt = construct_prompt(args, example) 

        start_time = time.time() 
        results = api_with_func_call(
            prompt=full_prompt, 
            max_tokens=args.max_tokens_per_call,
            temperature=args.temperature,
            n=args.n_sampling,
            top_p=args.top_p,
            executor=executor, 
            logger=logger 
        ) 
        
        end_time = time.time() 
        delta_time = end_time - start_time 
        logging(logger, "Total time for requesting the solution: " + format_time(delta_time) + "\n")

        # get prediction 
        predictions = [] 
        for r in results: 
            pred, report = run_execute(executor, r, args.prompt_type, execute=True) 
            predictions.append(pred) 
        
        sample = {'idx': idx, 'question': example['question'], 'pred': predictions, 'code': results} 

        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example: 
                sample[key] = example[key]
        try:
            writer.write(json.dumps(sample) + '\n')
            writer.flush() 
        except:
            logging(logger, ">>> Error writing to file\n") 
            continue 

    writer.close() 
    if logger: 
        logger.close() 


def get_total_gpt_response(args, raw_output_dir): 
    total_raw_output_file_path = os.path.join(raw_output_dir, 'total.jsonl') 

    # prepare_data_mumathcode
    examples = load_data_mumathcode(args.data_file) 

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

    num_samples = len(examples) 
    num_threads = args.num_threads 

    threads = [] 
    has_id = [] 

    if os.path.exists(total_raw_output_file_path): 
        return 

    delta = num_samples // num_threads 
    for i in range(num_threads): 
        start_index = i * delta 
        end_index = (i + 1) * delta if i < num_threads - 1 else num_samples 
        split_file_path = "zz_{}_{}.jsonl".format(start_index, end_index) 
        split_file_path = os.path.join(raw_output_dir, split_file_path) 

        if os.path.exists(split_file_path): 
            has_id.append(i) 

    for i in range(num_threads): 
        if i in has_id: 
            continue 
        start_index = i * delta 
        end_index = (i + 1) * delta if i < num_threads - 1 else num_samples 
        split_jsonl_data = examples[start_index: end_index]
        thread = threading.Thread(target=tread_processer, args=(args, split_jsonl_data, start_index, end_index, raw_output_dir), daemon=True) 

        threads.append(thread) 
        thread.start() 
        time.sleep(30*1) 

    for thread in threads: 
        thread.join() 
        
    total_raw_output_file = open(total_raw_output_file_path, 'w') 
    for i in range(num_threads): 
        start_index = i * delta 
        end_index = (i + 1) * delta if i < num_threads - 1 else num_samples 
        json_file_path = "zz_{}_{}.jsonl".format(start_index, end_index) 
        json_file_path = os.path.join(raw_output_dir, json_file_path) 
        if not os.path.exists(json_file_path): 
            total_raw_output_file.close() 
            os.remove(total_raw_output_file_path) 
            return 
        with open(json_file_path, 'r') as f: 
            for line in f.readlines(): 
                total_raw_output_file.write(line) 
    total_raw_output_file.close() 



def main(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--data_name", default="math", type=str)
    parser.add_argument("--data_file", type=str) 
    parser.add_argument("--model_name_or_path", default="gpt-4-tal-url", type=str) 
    parser.add_argument("--output_dir", type=str) 
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

    parser.add_argument("--num_threads", default=1, type=int) 
    parser.add_argument("--debug", action="store_true") 

    args = parser.parse_args() 
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm) 
    set_seed(args.seed) 

    raw_output_dir = os.path.join(args.output_dir, 'raw_new_data') 
    if not os.path.exists(raw_output_dir): 
        os.makedirs(raw_output_dir) 

    get_total_gpt_response(args, raw_output_dir) 


if __name__ == '__main__': 
    main() 
    


