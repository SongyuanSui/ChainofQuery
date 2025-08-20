import sys
import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm

from utils.load_data import *
from utils.myllm import MyChatGPT
from utils.database import MYSQLDB
from utils.helper import PipelineContext, AgentResult
from utils.pipeline import agent_pipeline
from utils.general_prompt import *
from utils.reasoner import chainofthought_answer_agent

def evaluator(generated_answer: str, answer: str) -> bool:
    if generated_answer.lower() == answer.lower():
        return True
    else:
        return False

global_dataset = None
global_data_process_func = None

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
        question = data_dict["question"]
        standard_answer = ", ".join(data_dict["answer"])
        tables = data_dict["tables"]
        sqldb = MYSQLDB(tables=tables)

        new_tables = sqldb.get_table_df()
        new_title = sqldb.get_table_title()
        table_dict = sqldb.get_table()

        log = {
                "sqls": [],
                "s_answer": "",
                "p_answer": "",
                "valid": None,
                "correct": None,
            }

        num_example_rows = 3
        prompt_table = create_table_prompt(df = new_tables, title = new_title)
        total_rows, prompt_rows = select_x_rows_prompt(full_table = False, df = new_tables, title = new_title, num_rows = num_example_rows)
        prompt_schema = prompt_table + prompt_rows

        my_llm = MyChatGPT(
            model_name=model_name,
            key=api_key
        )
        ctx = PipelineContext(
            llm = my_llm,
            sqldb = sqldb,
            question = question,
            prompt_schema = prompt_schema,
            title = new_title,
            previous_sql_query = None,
            total_rows = total_rows,
            log = log,
            flag = None,
            num_rows = num_example_rows,
            llm_options = None,
            debug = False,
            strategy = "top",
            extras = {}
        )

        valid_flag, answer_flag, sql_query, generated_answer, log = agent_pipeline(ctx, standard_answer)
        if valid_flag:
            log["valid"] = True
            predicted_answer = generated_answer
        else:
            log["valid"] = False
            if not answer_flag:
                analysis, predicted_answer = chainofthought_answer_agent(
                    llm = my_llm, question = question, table_dict = table_dict, debug = False
                )
            else:
                predicted_answer = generated_answer

        log["p_answer"] = predicted_answer
        log["s_answer"] = standard_answer

        if answer_flag:
            log["correct"] = True
        else:
            if evaluator(predicted_answer, standard_answer):
                log["correct"] = True
            else:
                log["correct"] = False

        sqldb.close()
        del sqldb

        return {
            "correct": log["correct"],
            "valid": log["valid"],
        }
    except Exception as e:
        print(f"[Error] Sample {i} failed: {e}", flush=True)
        return {
            "correct": False,
            "valid": False,
        }

def main():
    parser = argparse.ArgumentParser(description="Run Chain-of-Query experiments.")
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

    correct_count = sum(r["correct"] for r in results)
    invalid_sql = sum(not r["valid"] for r in results)

    print("Chain-of-Query:", correct_count, flush=True)
    print("Invalid:", invalid_sql, flush=True)

if __name__ == "__main__":
    main()