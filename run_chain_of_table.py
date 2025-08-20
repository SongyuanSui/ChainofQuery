import sys
import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from chain.utils.helper import *
from chain.utils.evaluate import *
from chain.utils.chain import *
from chain.operations import *
from chain.utils.llm import MyChatGPT
from utils.load_data import *

def chain_evaluator_wiki(generated_answer: str, answer: str) -> bool:
    if generated_answer.lower() == answer.lower():
        return True
    else:
        return False

def init_dataset():
    global global_dataset, global_data_process_func
    global_dataset, global_data_process_func = load_hg_dataset("wikitq")

def process_one_example_with_cfg(args):
    i, model_name, api_key = args
    return process_one_example(i, model_name, api_key)

def process_one_example(i, model_name, api_key):
    try:
        global global_dataset, global_data_process_func
        data = global_dataset["test"][i]
        data_dict = global_data_process_func(data)
        statement = data_dict["question"]
        answer = ", ".join(data_dict["answer"])
        table_caption = data_dict["tables"][0]['title']
        table_text = [data_dict["tables"][0]["table"]["header"]] + data_dict["tables"][0]["table"]["rows"]
        my_llm = MyChatGPT(
            model_name=model_name,
            key=api_key
        )
        sample = {
            "statement": statement,
            "table_caption": table_caption,
            "table_text": table_text,
            "cleaned_statement": statement,
            "chain": [],
        }

        proc_sample, dynamic_chain_log = dynamic_chain_exec_one_sample(
            sample=sample, llm=my_llm
        )
        output_sample = simple_query(
            sample=proc_sample,
            table_info=get_table_info(proc_sample),
            llm=my_llm,
            use_demo=True,
            llm_options=my_llm.get_model_options(
                temperature=0.0, per_example_max_decode_steps=200, per_example_top_p=1.0
            ),
        )
        act_chain_steps = []
        result = ""
        cotable_log = get_table_log(output_sample)
        for table_info in cotable_log:
            if table_info["act_chain"]:
                table_action = table_info["act_chain"][-1]
                if "skip" in table_action:
                    continue
                if "query" in table_action:
                    result = table_info["cotable_result"]
                act_chain_steps.append(table_action)
        answer_flag = chain_evaluator_wiki(generated_answer = result, answer = answer)
        return {
                "correct": answer_flag,
        }
    except Exception as e:
        print(f"[{i}] Fatal error: {e}")
        return {
            "correct": False,
        }

def main():
    parser = argparse.ArgumentParser(description="Run Chain-of-Table experiments.")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--num_samples", type=int, help="Number of test samples")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Please set OPENAI_API_KEY in your environment.")

    indices = list(range(args.num_samples))
    cfg_iter = ((i, args.model, api_key) for i in indices)
    results = []

    with Pool(processes=8, initializer=init_dataset) as pool:
        for res in tqdm(
            pool.imap_unordered(process_one_example_with_cfg, cfg_iter),
            total=len(indices),
        ):
            results.append(res)

    correct_count = sum(1 for r in results if r["correct"])
    print("Chain-of-Table:", correct_count, flush=True)

if __name__ == "__main__":
    main()