import pandas as pd
import json
import re
from _ctypes import PyObj_FromPtr
from typing import Any
from dataclasses import dataclass, field
from utils.myllm import MyChatGPT
from utils.database import MYSQLDB

@dataclass
class PipelineContext:
    '''
    Context for the pipeline.
    This context is passed to the agent and contains all the information needed to run the agent.
    It contains the following fields:
    - llm: The LLM to use for the agent.
    - sqldb: The database to use for the agent.
    - question: The question to ask the agent.
    - prompt_schema: The prompt schema to use for the agent.
    - title: The title of the agent.
    - previous_sql_query: The previous SQL query to use for the agent.
    - total_rows: The total number of rows in the database.
    - log: The log to use for the agent.
    - flag: A flag to indicate whether the certain clause is generated or not.
    - num_rows: The number of rows to use for the agent.
    - llm_options: The options to use for the LLM.
    - debug: Whether to use debug mode for the agent.
    - strategy: The strategy to use for the agent.
    - extras: Additional context for the agent.
    '''
    llm: MyChatGPT
    sqldb: MYSQLDB
    question: str
    prompt_schema: str
    title: str
    previous_sql_query: str | None
    total_rows: int
    log: dict
    flag: bool | None
    num_rows: int
    llm_options: dict | None
    debug: bool = False
    strategy: str = "top"
    extras: dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResult:
    flag_valid: bool
    next_agent: str | None
    updates: dict[str, object]

def add_prefix(sql: str) -> str:
    sql = sql.strip()
    if not sql.lower().startswith('select'):
        sql = 'SELECT ' + sql
    return sql

def table2df(table_text, num_rows=100):
    header, rows = table_text[0], table_text[1:]
    rows = rows[:num_rows]
    df = pd.DataFrame(data=rows, columns=header)
    return df

def table2string(
    table_text,
    num_rows=100,
    caption=None,
):
    df = table2df(table_text, num_rows)
    linear_table = ""
    if caption is not None:
        linear_table += "table caption : " + caption + "\n"
    header = "col : " + " | ".join(df.columns) + "\n"
    linear_table += header
    rows = df.values.tolist()
    for row_idx, row in enumerate(rows):
        row = [str(x) for x in row]
        line = "row {} : ".format(row_idx + 1) + " | ".join(row)
        if row_idx != len(rows) - 1:
            line += "\n"
        linear_table += line
    return linear_table

class NoIndent(object):
    def __init__(self, value):
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = "@@{}@@"
    regex = re.compile(FORMAT_SPEC.format(r"(\d+)"))

    def __init__(self, **kwargs):
        self.__sort_keys = kwargs.get("sort_keys", None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (
            self.FORMAT_SPEC.format(id(obj))
            if isinstance(obj, NoIndent)
            else super(MyEncoder, self).default(obj)
        )

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC
        json_repr = super(MyEncoder, self).encode(obj)
        for match in self.regex.finditer(json_repr):
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)
            json_repr = json_repr.replace(
                '"{}"'.format(format_spec.format(id)), json_obj_repr
            )
        return json_repr