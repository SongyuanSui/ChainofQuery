import re
import copy
import numpy as np
import pandas as pd

from utils.myllm import MyChatGPT
from utils.database import MYSQLDB
from utils.helper import PipelineContext, AgentResult
from utils.normalizer import convert_df_type, prepare_df_for_mysqldb_from_table
from utils.general_prompt import *

sufficiency_wikitq = '''[Instruction]
Your task is to decide whether the current SQL result is sufficient to answer the question.
The SQLite query retrieves information from a table to answer a given question.
Solve the task step by step if needed.
The table schema, a few example rows, the question, a SQLite query and its result will be provided.
[Constraints]
Answer "Yes" if and only if the current SQL result is sufficient to answer the question or with reasonable interpretation.
Otherwise, answer "No".
Base your decision solely on the given question, the provided SQLite query and its result.
The SQLite query does not need to be a complete or final answer.
If it includes most key information that allows you to infer the answer correctly, "Yes" is still acceptable, even if the question suggests that additional filtering, aggregation, or sorting might be needed.
Your response should be in this format:
Analysis:
**[Your analysis]**
Decision:
[Yes or No]
'''

answer_two_shot_example_wikitq = '''[Instruction]
Your task is to answer a question related to a given table based on the execution result attained by running SQLite.
Solve the task step by step if you need to.
The table schema, a few example rows, a SQLite query and the execution result will be provided.
Assume that you can always find the answer, so you must give an answer that makes sense to the question based on the given table.
Your answer should be as short as possible. Do not use sentences if one or two words will do.
[Response format] Your response should be in this format:
Analysis:
**[Your analysis]**
Answer:
[Your answer]

Table:
CREATE TABLE Fabrice_Santoro(
    row_id int,
    name text,
    _2001 text,
    _2002 text,
    _2003 text,
    _2004 text,
    _2005 text,
    _2006 text,
    _2007 text,
    _2008 text,
    _2009 text,
    _2010 text,
    career_nsr text,
    career_nwin_loss text)
/*
3 example rows:
SELECT * FROM Fabrice_Santoro LIMIT 3;
| row_id | name | _2001 | _2002 | _2003 | _2004 | _2005 | _2006 | _2007 | _2008 | _2009 | _2010 | career_nsr | career_nwin_loss |
| 0 | australian open | 2r | 1r | 3r | 2r | 1r | qf | 3r | 2r | 3r | 1r | 0 / 18 | 22-18 |
| 1 | french open | 4r | 2r | 2r | 3r | 1r | 1r | 1r | 2r | 1r | a | 0 / 20 | 17-20 |
| 2 | wimbledon | 3r | 2r | 2r | 2r | 2r | 2r | 2r | 1r | 2r | a | 0 / 14 | 11-14 |
*/
Question: did he win more at the australian open or indian wells?
SQLite:
SELECT name, career_nwin_loss FROM Fabrice_Santoro WHERE name LIKE "%australian%" OR name LIKE "%indian%";
Execution Result:
| name | career_nwin_loss |
| australian open | 22-18 |
| indian wells | 16-13 |
Output:
Analysis:
**At the Australian Open, his win-loss record is 22-18, giving him 22 wins.
At Indian Wells, his win-loss record is 16-13, giving him 16 wins.
22 > 16.**
Answer:
australian open

Table:
CREATE TABLE Playa_de_Oro_International_Airport(
    row_id int,
    rank int,
    city text,
    passengers text,
    ranking text,
    airline text)
/*
3 example rows:
SELECT * FROM Playa_de_Oro_International_Airport LIMIT 3;
| row_id | rank | city | passengers | ranking | airline |
| 0 | 1 | united states, los angeles | 14,749 | nan | alaska airlines |
| 1 | 2 | united states, houston | 5,465 | nan | united express |
| 2 | 3 | canada, calgary | 3,761 | nan | air transat, westjet |
*/
Question: how many more passengers flew to los angeles than to saskatoon from manzanillo airport in 2013?
SQLite:
SELECT city, passengers FROM Playa_de_Oro_International_Airport WHERE city LIKE "%los angeles%" OR city LIKE "%saskatoon%";
Execution Result:
| city | passengers |
| united states, los angeles | 14,749 |
| canada, saskatoon | 10,000 |
Output:
Analysis:
**Los Angeles had 14,749 passengers, and Saskatoon had 10,000 passengers.
The difference is 14,749 - 10,000 = 4,749.**
Answer:
4,749
'''

chainofthought_answer_two_shot_example_wikitq = '''[Instruction]
Your task is to answer a question related to a given table.
Solve the task step by step if you need to.
Assume you can always find the answer, so you must give an answer that makes sense to the question based on the given table.
Your answer should be as short as possible. Do not use sentences if one or two words will do.
[Response format] Your response should be in this format:
Analysis:
**[Your analysis]**
Answer:
[Your answer]

Table:
| row_id | name | _2001 | _2002 | _2003 | _2004 | _2005 | _2006 | _2007 | _2008 | _2009 | _2010 | career_nsr | career_nwin_loss |
| 0 | australian open | 2r | 1r | 3r | 2r | 1r | qf | 3r | 2r | 3r | 1r | 0 / 18 | 22-18 |
| 1 | french open | 4r | 2r | 2r | 3r | 1r | 1r | 1r | 2r | 1r | a | 0 / 20 | 17-20 |
| 2 | indian wells | 3r | 2r | 2r | 2r | 2r | 2r | 2r | 1r | 2r | a | 0 / 14 | 11-14 |
Question: did he win more at the australian open or indian wells?
Output:
Analysis:
**Compare the number of wins at the Australian Open and Indian Wells using the career_nwin_loss column.
Australian Open: 22 wins
Indian Wells: 11 wins**
Answer:
australian open

Table:
| row_id | rank | city | passengers | ranking | airline |
| 0 | 1 | united states, los angeles | 14,749 | nan | alaska airlines |
| 1 | 2 | united states, houston | 5,465 | nan | united express |
| 2 | 3 | canada, calgary | 10,000 | nan | air transat, westjet |
Question: how many more passengers flew to los angeles than to calgary from manzanillo airport in 2013?
Output:
Analysis:
**Los Angeles had 14,749 passengers, and Calgary had 10,000 passengers.
14,749 - 10,000 = 4,749 more passengers flew to Los Angeles.**
Answer:
4,749
'''

baseline_answer_two_shot_example_wikitq = '''Your task is to answer a question related to a given table.
Assume you can always find the answer, so you must give an answer that makes sense to the question based on the given table.
Your answer should be as short as possible. Do not use sentences if one or two words will do.
Output only the answer, with no explanation.

Table:
| row_id | name | _2001 | _2002 | _2003 | _2004 | _2005 | _2006 | _2007 | _2008 | _2009 | _2010 | career_nsr | career_nwin_loss |
| 0 | australian open | 2r | 1r | 3r | 2r | 1r | qf | 3r | 2r | 3r | 1r | 0 / 18 | 22-18 |
| 1 | french open | 4r | 2r | 2r | 3r | 1r | 1r | 1r | 2r | 1r | a | 0 / 20 | 17-20 |
| 2 | indian wells | 3r | 2r | 2r | 2r | 2r | 2r | 2r | 1r | 2r | a | 0 / 14 | 11-14 |
Question: did he win more at the australian open or indian wells?
Answer:
australian open

Table:
| row_id | rank | city | passengers | ranking | airline |
| 0 | 1 | united states, los angeles | 14,749 | nan | alaska airlines |
| 1 | 2 | united states, houston | 5,465 | nan | united express |
| 2 | 3 | canada, calgary | 10,000 | nan | air transat, westjet |
Question: how many more passengers flew to los angeles than to calgary from manzanillo airport in 2013?
Answer:
4,749
'''

sql_answer_two_shot_example_wikitq = '''[Instruction]
Your task is to answer a question related to a given table based on the execution result attained by running SQLite.
The table name, SQLite query and the execution result will be provided.
Assume that you can always find the answer, so you must give an answer that makes sense to the question based on the given table.
Your answer should be as short as possible. Do not use sentences if one or two words will do.
Output only the answer, with no explanation.

Table Name: Fabrice_Santoro
Question: did he win more at the australian open or indian wells?
SQLite:
SELECT name, career_nwin_loss FROM Fabrice_Santoro WHERE name LIKE "%australian%" OR name LIKE "%indian%";
Execution Result:
| name | career_nwin_loss |
| australian open | 22-18 |
| indian wells | 16-13 |
Answer:
australian open

Table Name: Playa_de_Oro_International_Airport
Question: how many more passengers flew to los angeles than to saskatoon from manzanillo airport in 2013?
SQLite:
SELECT city, passengers FROM Playa_de_Oro_International_Airport WHERE city LIKE "%los angeles%" OR city LIKE "%saskatoon%";
Execution Result:
| city | passengers |
| united states, los angeles | 14,749 |
| canada, saskatoon | 10,000 |
Answer:
4,749
'''

def _core_sufficiency_agent(llm: MyChatGPT, question: str, prompt_schema: str, title: str, sql_query: str, sql_result: dict,
                 log: dict, num_rows: int = 3, llm_options = None, debug = False, strategy="top") -> tuple[list, list, dict]:
    result_table = table2pipe(sql_result)
    prompt = sufficiency_wikitq + f"\nTable:\n{prompt_schema}Question: {question}\nSQLite:\n{sql_query}\nExecution Result:\n{result_table}Output:\n"

    temperature = 0.0
    n_sample = 1
    if llm_options is None:
        llm_options = llm.get_model_options(temperature = temperature, n_sample = n_sample, prompt = prompt)
    if debug:
        print("Final prompt:\n", prompt)
    response_list = llm.generate(prompt = prompt, options = llm_options, returnall = True)
    analysis_list = []
    answer_list = []
    for response in response_list:
        analysis, decision = extract_analysis_and_decision(response)
        analysis_list.append(analysis)
        answer_list.append(decision)

    return answer_list, analysis_list, log

def _evaluator(generated_answer: str, answer: str) -> bool:
    if generated_answer.lower() == answer.lower():
        return True
    else:
        return False

def _answers_evaluator(answer_list: list, standard_answer: str, log: dict) -> tuple[bool, str, dict]:
    if _evaluator(answer_list[0], standard_answer):
        log["p_answer"] = answer_list[0]
        return True, answer_list[0], log
    return False, answer_list[0], log

def _core_answer_agent(llm: MyChatGPT, question: str, prompt_schema: str, title: str, sql_query: str, sql_result: dict,
                 log: dict, num_rows: int = 3, llm_options = None, debug = False, strategy="top") -> tuple[list, list, dict]:
    result_table = table2pipe(sql_result)
    prompt = answer_two_shot_example_wikitq + f"\nTable:\n{prompt_schema}Question: {question}\nSQLite:\n{sql_query}\nExecution Result:\n{result_table}Output:\n"
    temperature = 0.0
    n_sample = 1
    if llm_options is None:
        llm_options = llm.get_model_options(temperature = temperature, n_sample = n_sample, prompt = prompt)
    if debug:
        print("Final prompt:\n", prompt)
    response_list = llm.generate(prompt = prompt, options = llm_options, returnall = True)
    analysis_list = []
    answer_list = []
    for response in response_list:
        analysis, answer = extract_analysis_and_answer(response)
        analysis_list.append(analysis)
        answer_list.append(answer)

    return answer_list, analysis_list, log

def chainofthought_answer_agent(llm: MyChatGPT, question: str, table_dict: dict, title: str = None,
                  llm_options = None, debug = False, strategy="top") -> tuple[str, str]:
    table_pipe = table2pipe(table_dict)
    prompt = chainofthought_answer_two_shot_example_wikitq + f"\nTable:\n{table_pipe}Question: {question}\nOutput:\n"
    if llm_options is None:
        llm_options = llm.get_model_options(prompt = prompt)
    if debug:
        print("Final prompt:\n", prompt)
    response = llm.generate(prompt = prompt, options = llm_options)
    analysis, answer = extract_analysis_and_answer(response)
    return analysis, answer

def baseline_answer_agent(llm: MyChatGPT, question: str, table_dict: dict, title: str = None,
                  llm_options = None, debug = False, strategy="top") -> str:
    table_pipe = table2pipe(table_dict)
    prompt = baseline_answer_two_shot_example_wikitq + f"\nTable:\n{table_pipe}Question: {question}\nOutput:\n"
    if llm_options is None:
        llm_options = llm.get_model_options(prompt = prompt)
    if debug:
        print("Final prompt:\n", prompt)
    response = llm.generate(prompt = prompt, options = llm_options)
    return response

def sql_answer_agent(llm: MyChatGPT, question: str, title: str, sql_query: str, sql_result: dict,
                    llm_options = None, debug = False, strategy="top") -> str:
    result_table = table2pipe(sql_result)
    prompt = sql_answer_two_shot_example_wikitq + f"\nTable Name: {title}\nQuestion: {question}\nSQLite:\n{sql_query}\nExecution Result:\n{result_table}Answer:\n"
    temperature = 0.0
    n_sample = 1
    if llm_options is None:
        llm_options = llm.get_model_options(temperature = temperature, n_sample = n_sample, prompt = prompt)
    if debug:
        print("Final prompt:\n", prompt)
    response_list = llm.generate(prompt = prompt, options = llm_options, returnall = True)
    analysis_list = []
    answer_list = []

    return response_list[0]

def SUFFICIENCY_agent(llm: MyChatGPT, sqldb: MYSQLDB, question: str, prompt_schema: str, title: str, standard_answer: str, sql_query: str,
                 log: dict, num_rows: int = 3, llm_options = None, debug = False, strategy="top") -> bool:
    sql_result = sqldb.execute_query(sql_query)
    predicted_answer_list, analysis_list, log = _core_sufficiency_agent(llm = llm, question = question, prompt_schema = prompt_schema, title = title,
                                                    sql_query = sql_query, sql_result = sql_result, log = log, debug = False)
    sufficient_flag = False
    if "yes" in predicted_answer_list[0].lower():
        sufficient_flag = True
    return sufficient_flag

def ANSWER_agent(llm: MyChatGPT, sqldb: MYSQLDB, question: str, prompt_schema: str, title: str, standard_answer: str, sql_query: str,
                 log: dict, num_rows: int = 3, llm_options = None, debug = False, strategy="top") -> tuple[bool, str, dict]:
    sql_result = sqldb.execute_query(sql_query)
    predicted_answer_list, analysis_list, log = _core_answer_agent(llm = llm, question = question, prompt_schema = prompt_schema, title = title,
                                                    sql_query = sql_query, sql_result = sql_result, log = log, debug = False)
    answer_flag = False
    answer_flag, generated_answer, log = _answers_evaluator(predicted_answer_list, standard_answer, log)
    return answer_flag, generated_answer, log