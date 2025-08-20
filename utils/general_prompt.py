import re
import copy
import numpy as np
import pandas as pd
import random
import ast

from utils.database import MYSQLDB
from utils.helper import PipelineContext, AgentResult
from utils.normalizer import convert_df_type, prepare_df_for_mysqldb_from_table

def _make_sqlite_friendly(name: str):
    if not name:
        return "_unnamed"
    name = re.sub(r'[^_a-zA-Z0-9]', '_', name)
    return name if name[0].isalpha() or name[0] == '_' else '_' + name

def extract_analysis_and_sql(reply: str) -> tuple[str, str]:
    analysis = ""
    sql = ""

    match_analysis = re.search(r"Analysis:\s*(.*?)\s*SQL:", reply, re.DOTALL | re.IGNORECASE)
    if match_analysis:
        analysis = match_analysis.group(1).strip()

    match_sql = re.search(r"```sql\s*(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    if match_sql:
        sql = match_sql.group(1).strip()
    else:
        match_sql = re.search(r"SQL:\s*(.*)", reply, re.DOTALL | re.IGNORECASE)
        if match_sql:
            sql = match_sql.group(1).strip()
    return analysis, sql

def extract_analysis_and_decision(reply: str) -> tuple[str, str]:
    analysis = ""
    decision = ""

    match_analysis = re.search(r"Analysis:\s*(.*?)\s*Decision:", reply, re.DOTALL | re.IGNORECASE)
    if match_analysis:
        analysis = match_analysis.group(1).strip()

    match_decision = re.search(r"Decision:\s*([^\n\r]*)", reply, re.DOTALL | re.IGNORECASE)
    if match_decision:
        decision = match_decision.group(1).strip()

    return analysis, decision

def extract_analysis_and_answer(reply: str) -> tuple[str, str]:
    analysis = ""
    answer = ""

    match_analysis = re.search(r"Analysis:\s*(.*?)\s*Answer:", reply, re.DOTALL | re.IGNORECASE)
    if match_analysis:
        analysis = match_analysis.group(1).strip()

    match_answer = re.search(r"Answer:\s*([^\n\r]*)", reply, re.DOTALL | re.IGNORECASE)
    if match_answer:
        answer = match_answer.group(1).strip()

    return analysis, answer

def extract_sql(reply: str) -> str:
    sql = ""
    match_sql = re.search(r"```sql\s*(.*?)```", reply, re.DOTALL | re.IGNORECASE)
    if match_sql:
        sql = match_sql.group(1).strip()
    else:
        match_sql = re.search(r"SQL:\s*(.*)", reply, re.DOTALL | re.IGNORECASE)
        if match_sql:
            sql = match_sql.group(1).strip()
    return sql

def ensure_strings(lst):
    return [str(item) if not isinstance(item, str) else item for item in lst]

def table2pipe(table: dict) -> str:
    try:
        pipe_table = "| " + " | ".join(table["header"]) + " |\n"
        for raw_row in table["rows"]:
            row = ensure_strings(raw_row)
            pipe_row = "| " + " | ".join(row) + " |\n"
            pipe_table += pipe_row
        return pipe_table
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError("Wrong table format.") from e

def contains_yes(s: str) -> bool:
    return "yes" in s.lower()

def remove_semicolon(original: str) -> str:
    if not original.endswith(";"):
        raise ValueError("SQL must end with ';'")

    new = original[:-1]
    return new

def create_table_prompt(df: pd.DataFrame, title: str) -> str:
    prompt = "CREATE TABLE {}(\n".format(title)
    for header in df.columns:

        column_type = 'text'
        try:
            if df[header].dtype == 'int64':
                column_type = 'int'
            elif df[header].dtype == 'float64':
                column_type = 'real'
            elif df[header].dtype == 'datetime64':
                column_type = 'datetime'
        except AttributeError as e:
            pass

        prompt += '\t{} {},\n'.format(header, column_type)
    prompt = prompt.rstrip(',\n') + ')\n'
    return prompt

def select_x_rows_prompt(full_table: bool, df: pd.DataFrame, title: str, num_rows: int = 3) -> tuple[int, str]:
    total_number = len(df)
    if full_table:
        prompt = f'/*\nAll rows of the table:\nSELECT * FROM {title};\n'
        num_rows = total_number
    else:
        num_rows = min(num_rows, total_number)
        if num_rows == total_number:
            prompt = f'/*\nAll rows of the table:\nSELECT * FROM {title};\n'
            num_rows = total_number
        else:
            prompt = f'/*\n{num_rows} example rows:\nSELECT * FROM {title} LIMIT {num_rows};\n'

    prompt += "| "
    prompt += " | ".join(df.columns)
    prompt += " |\n"

    for _, row in df.iloc[:num_rows].iterrows():
        prompt += "| "
        prompt += " | ".join(map(str, row.values))
        prompt += " |\n"

    prompt += '*/\n'
    return total_number, prompt

def _select_temp_x_rows_prompt(full_table: bool, sqldb: MYSQLDB, title: str, total_num: int, num_rows: int = 3) -> str:
    if full_table:
        sql_rows = f"SELECT * FROM {title};"
        prompt = f'/*\nAll rows of the table:\nSELECT * FROM {title};\n'
        num_rows = total_num
    else:
        num_rows = min(num_rows, total_num)
        if num_rows == total_num:
            sql_rows = f"SELECT * FROM {title};"
            prompt = f"/*\nAll rows of the table:\nSELECT * FROM {title};\n"
            num_rows = total_num
        else:
            sql_rows = f"SELECT * FROM {title} LIMIT {num_rows};"
            prompt = f"/*\n{num_rows} example rows:\nSELECT * FROM {title} LIMIT {num_rows};\n"
    execute_result = sqldb.execute_query(sql_rows)
    if not execute_result["sqlite_error"]:
        header = execute_result["header"]
        rows = execute_result["rows"]
        prompt += "| "
        prompt += " | ".join(header)
        prompt += " |\n"
        for row_list in rows:
            row = ensure_strings(row_list)
            prompt += "| "
            prompt += " | ".join(row)
            prompt += " |\n"
        prompt += '*/\n'
        return prompt
    else:
        raise ValueError("The temp table rows query failed.")

def TEMP_table_prompt(ctx: PipelineContext) -> AgentResult:
    sqldb = ctx.sqldb
    sql_create = ctx.previous_sql_query
    title = ctx.title
    log = ctx.log

    try:
        cte_names = re.findall(r'\bCREATE TABLE\s+([^\s]+)\s+AS\b', sql_create, re.IGNORECASE | re.DOTALL)
        if not cte_names:
            table_name = "table_name_tmp"
        else:
            first_cte_name = cte_names[0].split(',')[0].strip()
            table_name = _make_sqlite_friendly(first_cte_name)
        temp_table_schema = sqldb.get_table_schema(table_name)
        total_num = temp_table_schema[0]
        temp_schema = temp_table_schema[1]
        prompt = "CREATE TABLE {}(\n".format(table_name)
        for column in temp_schema:
            column_name = column[0]
            column_type = column[1]
            prompt += '\t{} {},\n'.format(column_name, column_type)
        prompt = prompt.rstrip(',\n') + ')\n'
        row_prompt = _select_temp_x_rows_prompt(full_table = False, sqldb = sqldb, title = table_name, total_num = total_num, num_rows = 3)
        prompt += row_prompt
        return AgentResult(
            flag_valid = True,
            next_agent = "Select",
            updates = {
                "sql": sql_create,
                "log": log,
                "table_name": table_name,
                "total_number": total_num,
                "prompt": prompt
            }
        )
    except Exception as e:
        return AgentResult(
            flag_valid = False,
            next_agent = "Select",
            updates = {
                "sql": sql_create,
                "log": log,
                "error": str(e),
                "error_class": type(e).__name__
            }
        )