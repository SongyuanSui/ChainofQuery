import sys
import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import pandas as pd
import time
import traceback

from utils.myllm import MyChatGPT
from utils.database import MYSQLDB
from utils.load_data import *
from utils.agents.base_sql_generator import base_sql_agent
from utils.reasoner import sql_answer_agent, baseline_answer_agent
from utils.general_prompt import *

def basic_sql_evaluator_wiki(generated_answer: str, answer: str) -> bool:
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
    global global_dataset, global_data_process_func

    data = global_dataset["test"][i]
    data_dict = global_data_process_func(data)
    result = {
        "idx": i,
        "question": "",
        "sql_pre": None,
        "sql_flag": False,
        "answer_flag": False,
    }

    try:
        question = data_dict["question"]
        standard_answer = ", ".join(data_dict["answer"])
        tables = data_dict["tables"]
        sqldb = MYSQLDB(tables=tables)

        my_llm = MyChatGPT(
            model_name=model_name,
            key=api_key
        )

        new_tables = sqldb.get_table_df()
        new_title = sqldb.get_table_title()
        table_dict = sqldb.get_table()

        num_example_rows = 3
        prompt_table = create_table_prompt(df = new_tables, title = new_title)
        total_rows, prompt_rows = select_x_rows_prompt(full_table = False, df = new_tables, title = new_title, num_rows = num_example_rows)
        prompt_schema = prompt_table + prompt_rows
        sql_flag = False
        answer_flag = False

        sql_query = base_sql_agent(llm = my_llm, question = question, prompt_schema = prompt_schema, title = new_title)

        if sql_query:
            sql_res = sqldb.execute_query(sql_query)
            if not sql_res["sqlite_error"]:
                pre_answer = sql_answer_agent(
                    llm=my_llm,
                    question=question,
                    title=new_title,
                    sql_query=sql_query,
                    sql_result=sql_res
                )
                answer_flag = basic_sql_evaluator_wiki(pre_answer, standard_answer)
                sql_flag = True
            else:
                pre_answer = baseline_answer_agent(
                    llm = my_llm, question = question, table_dict = table_dict, debug = False
                )
                answer_flag = basic_sql_evaluator_wiki(pre_answer, standard_answer)
        else:
            pre_answer = baseline_answer_agent(
                    llm = my_llm, question = question, table_dict = table_dict, debug = False
                )
            answer_flag = basic_sql_evaluator_wiki(pre_answer, standard_answer)

        result.update({
            "question": question,
            "sql_pre": sql_query,
            "sql_flag": sql_flag,
            "answer_flag": answer_flag
        })

        sqldb.close()
        del sqldb

        return result

    except Exception as e:
        print(f"[ERROR] Sample {i} failed with error: {e}", flush=True)
        traceback.print_exc()
        time.sleep(2)
        return {
            "idx": i,
            "question": "",
            "sql_pre": None,
            "sql_flag": False,
            "answer_flag": False,
        }

def main():
    parser = argparse.ArgumentParser(description="Run Basic SQL experiments.")
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

    correct_count = sum(1 for r in results if r["answer_flag"])
    invalid_sql = sum(1 for r in results if not r["sql_flag"])

    print("Basic SQL:", correct_count, flush=True)
    print("Invalid:", invalid_sql, flush=True)

if __name__ == "__main__":
    main()