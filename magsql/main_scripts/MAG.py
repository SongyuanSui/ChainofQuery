# -*- coding: utf-8 -*-
import pandas as pd
import json
from func_timeout import func_timeout,FunctionTimedOut
from magsql.main_scripts.utils import parse_json, check_letter, contain_value, add_prefix, load_json_file, extract_world_info, is_email, is_valid_date_column, extract_sql, detect_special_char, add_quotation_mark, get_matched_content_sequence, get_chosen_schema, extract_subquery
from magsql.main_scripts.bridge_content_encoder import get_matched_entries

from magsql.main_scripts.const import *
from typing import List
from copy import deepcopy

import sqlite3
import time
import abc
import sys
import os
import pandas as pd
from tqdm import tqdm, trange


class BaseAgent(metaclass=abc.ABCMeta):
    # Define a base class
    def __init__(self):
        pass

    @abc.abstractmethod
    def talk(self, message: dict):
        pass


class Soft_Schema_linker(BaseAgent):
    """
    Get database description and then extract entities from questions before selecting related schema
    """
    name = SCHEMALINKER_NAME
    description = "Get database description and then extract entities from questions before selecting related schema"

    def __init__(self, llm, without_selector: bool = False):
        super().__init__()
        self.llm = llm
        self.without_selector = without_selector

        self.match_dict = {}
        self.total_content_dict = {}
        self._message = {}

        self.db_summary = {}


    def _get_column_attributes(self, sqldb, table_name: str):
        """
        Extract column names and types from a table via sqldb (instead of raw sqlite3 cursor).
        """
        try:
            _, schema = sqldb.get_table_schema(table_name)
            column_names = [col_name for col_name, _ in schema]
            column_types = [col_type for _, col_type in schema]
            return column_names, column_types
        except Exception as e:
            print(f"[Error] Failed to extract column attributes for table {table_name}: {e}")
            return [], []

    def _data_prematch(self, sqldb, question_id: str, question: str, evidence: str) -> list:
        table_name = sqldb.get_table_title()
        all_column_names, all_column_types = self._get_column_attributes(sqldb, table_name)

        if self.total_content_dict.get(table_name) is None:
            self.total_content_dict[table_name] = {}
            for col_name, col_type in zip(all_column_names, all_column_types):
                if col_type.upper() != "TEXT":
                    continue

                rows = sqldb.get_rows(table_name)
                col_idx = all_column_names.index(col_name)
                values = [r[col_idx] for r in rows if r[col_idx] not in (None, '')]
                values = list(dict.fromkeys(values))

                check = any((not isinstance(v, str) or not v.isdecimal()) for v in values if v is not None)
                if not check:
                    continue
                self.total_content_dict[table_name][col_name] = values

        inputs = question
        matched_list = []
        for col_name, contents in self.total_content_dict[table_name].items():
            if len(contents) > 2000:
                continue
            fm_contents = get_matched_entries(inputs, contents)
            if fm_contents is None:
                continue
            for _match_str, (field_value, _s_match_str, match_score, s_match_score, _match_size) in fm_contents:
                if match_score < 0.9:
                    continue
                if field_value.isdecimal() or len(field_value) == 1:
                    continue
                matched_list.append(f"{table_name}.`{col_name}` = '{field_value}'")
        matched_list = list(set(matched_list))
        matched_list_top10 = sorted(matched_list, key=lambda i: len(i), reverse=True)[:10]
        return matched_list_top10

    def _build_part_bird_table_schema_list_str(self, table_name, new_columns_desc, new_columns_val, db_id=None):
        # Schema representation without detailed descriptions or value examples
        schema_desc_str = f"# {table_name}: "
        extracted_column_infos = []

        for (col_name, _, col_type, _), _ in zip(new_columns_desc, new_columns_val):
            col_line_text = f"{col_name} ({col_type})"
            extracted_column_infos.append(col_line_text)

        schema_desc_str += '[' + ', '.join(extracted_column_infos) + ']' + '\n'
        return schema_desc_str

    def _build_total_bird_table_schema_list_str(self, table_name, new_columns_desc, new_columns_val, db_id=None):
        # Complete schema representation with full column names and value examples
        schema_desc_str = f"# Table: {table_name}\n"
        extracted_column_infos = []

        for (col_name, full_col_name, col_type, extra_desc), (_, col_values_str) in zip(new_columns_desc, new_columns_val):
            extra_desc = f"And {extra_desc}" if extra_desc else ""
            extra_desc = extra_desc[:100]

            col_line_text = f"  ({col_name} <{col_type}>,"
            if full_col_name:
                col_line_text += f" {full_col_name.strip()}."
            if col_values_str:
                col_line_text += f" Value examples: {col_values_str}."
            if extra_desc:
                col_line_text += f" {extra_desc}"
            col_line_text += ")"
            extracted_column_infos.append(col_line_text)

        schema_desc_str += '[\n' + ',\n'.join(extracted_column_infos) + '\n]' + '\n'
        return schema_desc_str


    def _get_related_details(self, table_name, new_columns_desc, new_columns_val, extracted_schema):
        """
        Generate detailed column descriptions based on extracted schema selection,
        column names, types, and value examples. No external description files are used.
        """
        related_details = ''
        llm_chosen_columns = extracted_schema.get(table_name, [])
        llm_chosen_columns = [col.strip('`') for col in llm_chosen_columns]

        for (col_name, full_col_name, col_type, col_extra_desc), (_, col_values_str) in zip(new_columns_desc, new_columns_val):
            if col_name in llm_chosen_columns:
                col_details = f"{table_name}.`{col_name}`: "

                if full_col_name:
                    col_details += f"The column '{col_name}' in Table <{table_name}> has description \"{full_col_name.strip()}\". "

                if col_values_str:
                    col_details += f"Value examples: {col_values_str}. "

                if col_extra_desc and str(col_extra_desc).lower() != 'nan':
                    col_details += f"{col_extra_desc}"

                related_details += col_details.strip() + "\n"

        return related_details


    def _get_db_desc_str(self,
                         sqldb,
                         table_name: str,
                         extracted_schema: dict,
                         matched_content_dict: dict = None,
                         complete: bool = True) -> List[str]:
        """
        Add foreign keys, and value descriptions of focused columns.
        :param db_id: name of sqlite database
        :param extracted_schema: {table_name: "keep_all" or "drop_all" or ['col_a', 'col_b']}
        :return: Detailed columns info of db; foreign keys info of db
        """
        header = sqldb.get_header()
        rows = sqldb.get_rows(table_name)
        row_count, schema = sqldb.get_table_schema(table_name)
        pk_cols = sqldb.get_primary_keys(table_name)

        pk_cols = sqldb.get_primary_keys(table_name) if hasattr(sqldb, 'get_primary_keys') else []

        # Column Value Example Extraction
        value_examples = {}
        for col_idx, col_name in enumerate(header):
            values = [r[col_idx] for r in rows if r[col_idx] not in (None, '')]
            values = list(dict.fromkeys(values))  # remove duplicates
            if len(values) > 6:
                values = values[:6]
            value_examples[col_name] = str(values).replace("\n", " ")

        # Construct column information [(col_name, full_col_name, col_type, extra_desc)], [(col_name, value_example_str)]
        new_columns_desc = []
        new_columns_val = []
        for col_name, col_type in schema:
            full_col_name = col_name.replace("_", " ")
            extra_desc = ''
            new_columns_desc.append((col_name, full_col_name, col_type, extra_desc))
            new_columns_val.append((col_name, value_examples.get(col_name, '')))

        # Construct schema description
        if complete:
            schema_desc_str = self._build_total_bird_table_schema_list_str(table_name, new_columns_desc, new_columns_val, db_id=None)
        else:
            schema_desc_str = self._build_part_bird_table_schema_list_str(table_name, new_columns_desc, new_columns_val, db_id=None)

        # Constructs additional details (such as the fields used)
        related_details_str = self._get_related_details(table_name, new_columns_desc, new_columns_val, extracted_schema)

        # Primary key information
        pk_desc_str = f"{table_name}.`{pk_cols[0]}`" if pk_cols else ""

        # Foreign key is empty
        fk_desc_str = ""

        # Schema column selection results
        chosen_db_schem_dict = {table_name: [col[0] for col in new_columns_desc]}

        # match_content string
        match_desc_str = get_matched_content_sequence(matched_content_dict or {})

        return schema_desc_str.strip(), fk_desc_str, pk_desc_str, chosen_db_schem_dict, match_desc_str, related_details_str

    def _get_summary(self, sqldb, table_name):
        """
        Summarize the structure of a single table using its schema.
        Used for TableQA tasks with one table per example.
        """
        db_schema, db_fk, db_pk, chosen_db_schem_dict, match_content, column_details = self._get_db_desc_str(
            sqldb=sqldb,
            table_name=table_name,
            extracted_schema={},
            complete=True
        )

        prompt = summarizer_template.format(db_id=table_name, desc_str=db_schema)
        summary_json = self.llm.generate(prompt = prompt)
        summary_dict = parse_json(summary_json)

        temp_str = ''
        for tb_name, tb_summary in (summary_dict or {}).items():
            temp_str += f"# {tb_name}: {tb_summary}\n"
        return {table_name: temp_str.strip()}


    def _is_need_prune(self, sqldb, table_name: str, db_schema: str) -> bool:
        """
        Decide whether pruning is needed based on number of columns or total tokens in schema.
        """
        try:
            # Option 1: Use token count of db_schema
            # encoder = tiktoken.get_encoding("cl100k_base")
            # tokens = encoder.encode(db_schema)
            # return len(tokens) >= 2500  # Lower threshold for single-table case

            # Option 2: Use structural heuristics
            row_count, schema = sqldb.get_table_schema(table_name)
            total_column_count = len(schema)

            # Thresholds can be tuned
            if total_column_count <= 10:
                return False
            else:
                # import ipdb; ipdb.set_trace()
                return True
        except Exception as e:
            print(f"[Prune Check Error] Failed for table {table_name}: {e}")
            return False  # Fallback: do not prune

    def _prune(self,
                sqldb,
                table_name: str,
                query: str,
                db_schema: str,
                db_pk: str,
                db_fk: str,
                evidence: str = None,
                summary_str: str = None,
                matched_list: list = []) -> dict:
        """
        Prune the table schema using LLM based on query, evidence, schema info, etc.
        """
        if matched_list:
            matched_str = '; '.join(matched_list)
        else:
            matched_str = 'No matched values.'

        # Build pruning prompt without db_summary
        prompt = schema_linker_template.format(
            db_id=table_name,
            query=query,
            evidence=evidence,
            desc_str=db_schema,
            fk_str=db_fk,
            pk_str=db_pk,
            matched_list=matched_str,
            summary_str=summary_str
        )

        # import ipdb; ipdb.set_trace()
        # print(prompt)
        reply = self.llm.generate(prompt = prompt)
        # print(reply)

        extracted_schema_dict = get_chosen_schema(parse_json(reply))
        return extracted_schema_dict

    def talk(self, message: dict):
        """
        Soft_Schema_linker agent entry point for TableQA.
        Message must contain:
            - sqldb: an instance of MYSQLDB
            - table_title: name of the table
            - query: user question
            - extracted_schema: optional schema if provided
        Output keys added to message:
            - desc_str, fk_str, pk_str, match_content_str, columns_details_str, etc.
        """
        # import ipdb; ipdb.set_trace()
        if message['send_to'] != self.name:
            return

        self._message = message

        sqldb = message['sqldb']
        table_name = message['table_title']
        query = message['query']
        ext_sch = message.get('extracted_schema', {})
        evidence = message.get('evidence', '')
        idx = message.get('idx', 0)

        # Match dict lookup (if enabled)
        idx_str = str(idx)
        if self.match_dict.get(idx_str) is None:
            matched_list = self._data_prematch(sqldb, question_id=idx_str, question=query, evidence=evidence)
            self.match_dict[idx_str] = matched_list
        else:
            matched_list = self.match_dict[idx_str]
        # import ipdb; ipdb.set_trace()
        message['matched_list'] = matched_list

        # Generate full schema description
        db_schema, db_fk, db_pk, chosen_db_schem_dict, match_content, column_details = self._get_db_desc_str(
            sqldb=sqldb,
            table_name=table_name,
            extracted_schema=ext_sch,
            matched_content_dict=None,
            complete=True
        )

        # self.db_summary = self._get_summary(sqldb, table_name)
        # summary_str = self.db_summary.get(table_name, "")
        summary_str = ""

        message['complete_desc_str'] = db_schema
        message['summary_str'] = summary_str
        need_prune = self._is_need_prune(sqldb, table_name, db_schema)

        if self.without_selector:
            need_prune = False

        if ext_sch == {} and need_prune:
            # import ipdb; ipdb.set_trace()
            raw_extracted_schema_dict = self._prune(
                sqldb=sqldb,
                table_name=table_name,
                query=query,
                db_schema=db_schema,
                db_fk=db_fk,
                db_pk=db_pk,
                evidence=evidence,
                matched_list=matched_list,
                summary_str=summary_str
            )

            # Re-generate schema description after pruning
            db_schema, db_fk, db_pk, chosen_db_schem_dict, match_content, column_details = self._get_db_desc_str(
                sqldb=sqldb,
                table_name=table_name,
                extracted_schema=raw_extracted_schema_dict,
                matched_content_dict=None,
                complete=False
            )

            message.update({
                'extracted_schema': raw_extracted_schema_dict,
                'chosen_db_schem_dict': chosen_db_schem_dict,
                'desc_str': db_schema,
                'fk_str': db_fk,
                'pk_str': db_pk,
                'pruned': True,
                'match_content_str': match_content,
                'columns_details_str': column_details,
                'send_to': DECOMPOSER_NAME
            })
        else:
            message.update({
                'chosen_db_schem_dict': chosen_db_schem_dict,
                'desc_str': db_schema,
                'fk_str': db_fk,
                'pk_str': db_pk,
                'pruned': False,
                'match_content_str': match_content,
                'send_to': DECOMPOSER_NAME
            })


class Decomposer(BaseAgent):
    """
    Decompose the question into Targets and Conditions, and then splice them one by one to get a series of Sub-questions
    """
    name = DECOMPOSER_NAME
    description = "Decompose the question into Targets and Conditions, and then splice them one by one to get a series of Sub-questions"

    def __init__(self, llm):
        super().__init__()
        self._message = {}
        self.llm = llm

    def talk(self, message: dict):
        """
        :param message: {"query": user_query,
                         "evidence": extra_info,
                         "desc_str": description of db schema,
                         "fk_str": foreign keys of database}
        :return: decompose question into sub ones and solve them in generated SQL
        """
        # import ipdb; ipdb.set_trace()
        if message['send_to'] != self.name:
            return

        self._message = message
        query = message.get('query')
        evidence = message.get('evidence')

        prompt = pure_decomposer_template.format(query=query, evidence=evidence)
        reply = self.llm.generate(prompt = prompt)
        # import ipdb; ipdb.set_trace()
        # print(reply)

        reply_list = extract_subquery(reply)
        if not reply_list:
            reply_list.append(query)

        message['subquery_list'] = reply_list
        # Increase fault tolerance by replacing the last sub-question with the original question
        message['subquery_list'][-1] = query
        message['initial_state'] = True
        message['send_to'] = GENERATOR_NAME


class Generator(BaseAgent):
    """
    Generate Sub-SQL iteratively using CoT
    """
    name = GENERATOR_NAME
    description = "Generate Sub-SQL iteratively using CoT"

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self._message = {}

    def talk(self, message: dict):
        """
        :param message: includes question, schema, evidence, subqueries, etc.
        :return: generates SQL based on current sub-question and context
        """
        # import ipdb; ipdb.set_trace()
        if message['send_to'] != self.name:
            return

        self._message = message
        evidence = message.get('evidence')
        schema_info = message.get('desc_str')
        fk_info = message.get('fk_str')
        pk_info = message.get('pk_str')
        matched_list = message.get('matched_list')
        subqueries = message.get('subquery_list')
        column_details = message.get('columns_details_str')

        matched_str = '; '.join(matched_list) if matched_list else 'No matched values.'
        focus_query = subqueries[0]
        # import ipdb; ipdb.set_trace()
        initial = message['initial_state']

        # First round vs subsequent round
        if initial:
            # import ipdb; ipdb.set_trace()
            prompt = soft_schema_initial_generator_template.format(
                query=focus_query,
                evidence=evidence,
                desc_str=schema_info,
                fk_str=fk_info,
                pk_str=pk_info,
                detailed_str=column_details,
                matched_list=matched_str
            )
            message['initial_state'] = False
        else:
            # import ipdb; ipdb.set_trace()
            subquery = message['last_subquery']
            subsql = message['sub_sql']
            prompt = soft_schema_continuous_generator_template.format(
                query=focus_query,
                evidence=evidence,
                desc_str=schema_info,
                fk_str=fk_info,
                pk_str=pk_info,
                detailed_str=column_details,
                subquery=subquery,
                subsql=subsql,
                matched_list=matched_str
            )

        # import ipdb; ipdb.set_trace()
        # print(prompt)
        reply = self.llm.generate(prompt = prompt)
        # print(reply)
        # import ipdb; ipdb.set_trace()
        sql_statement = extract_sql(reply)
        message['old_chain_of_thoughts'] = reply
        message['final_sql'] = sql_statement
        message['fixed'] = False
        message['send_to'] = REFINER_NAME


class Refiner(BaseAgent):
    name = REFINER_NAME
    description = "Execute SQL and preform validation"

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self._message = {}

    def _execute_sql(self, sql: str, sqldb) -> dict:
        try:
            result = func_timeout(30, sqldb.execute_query, args=(sql,))
            return {
                "sql": str(sql),
                "data": result["rows"],
                "sqlite_error": result["sqlite_error"],
                "exception_class": result["exception_class"]
            }
        except FunctionTimedOut as te:
            return {
                "sql": str(sql),
                "sqlite_error": str(te.args),
                "exception_class": str(type(te).__name__)
            }
        except Exception as e:
            return {
                "sql": str(sql),
                "sqlite_error": str(e.args),
                "exception_class": str(type(e).__name__)
            }


    @staticmethod
    def _is_need_refine(exec_result: dict, try_times: int) -> bool:
        data = exec_result.get('data', None)

        if try_times >= 2:
            return False

        if data is None:
            return True

        if len(data) == 0:
            exec_result['sqlite_error'] = 'no data selected'
            return True

        if len(data) == 1:
            return False

        check_all_None = True
        check_has_None = False
        for row in data:
            for value in row:
                if value is None:
                    check_has_None = True
                else:
                    check_all_None = False

        if check_has_None and not check_all_None:
            exec_result['sqlite_error'] = 'exist None value, you can add `IS NOT NULL` in SQL'
            return True

        return False

    # No call?
    # def _value_retriver(self, target: str, db_id: str, db_content_dict: dict, related_schema: dict):
    #     inputs = target.strip('\'').strip('%')
    #     matched_list = []
    #     for tb_name, contents_dict in db_content_dict.items():
    #         related_cols = related_schema.get(tb_name,[])
    #         for col_name,contents in contents_dict.items():
    #             if col_name in related_cols:
    #                 matched_contents = []
    #                 for v in contents:
    #                     if v != None:
    #                         if inputs.lower().replace(" ","") in v.lower().replace(" ","") and len(v) <= 2 + len(inputs) and v != inputs:
    #                             matched_contents.append(v)
    #                 for match_str in matched_contents:
    #                     matched_list.append(f"{tb_name}.`{col_name}` = \'" + match_str + "\'")
    #     matched_list = list(set(matched_list))
    #     matched_list_top5 = sorted(matched_list,key = lambda i:len(i),reverse=False)[:4]
    #     return matched_list_top5

    # def _judge_value(self, sql: str):
    #     values_targets = []
    #     value_list = contain_value(sql)
    #     if value_list != None:
    #         for item in value_list:
    #             if check_letter(item):
    #                 values_targets.append(item)
    #     return values_targets


    def _refine(self,
                sqldb,
                query: str,
                evidence: str,
                schema_info: str,
                pk_info: str,
                fk_info: str,
                column_details: str,
                error_info: dict,
                complete_schema: str,
                matched_content: str) -> dict:

        sql_arg = add_prefix(error_info.get('sql'))
        sqlite_error = error_info.get('sqlite_error')
        exception_class = error_info.get('exception_class')

        # Add explanation for common "no such column" error
        if "no such column" in sqlite_error.lower():
            sqlite_error += " (Check if the column in the SQL is selected from the correct table based on the 【Database info】 at first, and then check if the column name is enclosed in backticks.)"
            prompt = refiner_template.format(
                query=query,
                evidence=evidence,
                desc_str=complete_schema,
                pk_str=pk_info,
                detailed_str=column_details,
                fk_str=fk_info,
                sql=sql_arg,
                sqlite_error=sqlite_error,
                exception_class=exception_class
            )
            filter_error = True

        # Use nested refiner when SELECTs are nested and return no data
        elif "no data selected" in sqlite_error.lower() and sql_arg.count('SELECT') > 1:
            prompt = nested_refiner_template.format(
                query=query,
                evidence=evidence,
                desc_str=schema_info,
                pk_str=pk_info,
                detailed_str=column_details,
                fk_str=fk_info,
                sql=sql_arg,
                sqlite_error=sqlite_error,
                matched_content=matched_content,
                exception_class=exception_class
            )
            # print(prompt)
            filter_error = False

        # All other SQL errors
        else:
            prompt = refiner_template.format(
                query=query,
                evidence=evidence,
                desc_str=schema_info,
                pk_str=pk_info,
                detailed_str=column_details,
                fk_str=fk_info,
                sql=sql_arg,
                sqlite_error=sqlite_error,
                exception_class=exception_class
            )
            filter_error = False

        # Call LLM
        reply = self.llm.generate(prompt=prompt)
        # import ipdb; ipdb.set_trace()
        # print(reply)
        res = extract_sql(reply)
        return res, filter_error


    def talk(self, message: dict):
        """
        Execute SQL and perform validation.
        :param message: {
            "query": user_query,
            "evidence": extra_info,
            "desc_str": schema info,
            "fk_str": foreign keys,
            "final_sql": SQL to be verified,
            "sqldb": database interface object
        }
        :return: refine SQL if needed
        """
        # import ipdb; ipdb.set_trace()
        if message['send_to'] != self.name:
            return

        self._message = message

        sqldb = message['sqldb']
        old_sql = message.get('pred', message.get('final_sql'))
        query = message.get('subquery_list')[0]
        evidence = message.get('evidence')
        schema_info = message.get('desc_str')
        fk_info = message.get('fk_str')
        complete_schema = message.get('complete_desc_str')
        pk_info = message.get('pk_str')
        column_details = message.get('columns_details_str')

        matched_list = message.get('matched_list') or []
        matched_content = '; '.join(matched_list) if matched_list else 'No matched values.'

        try_times = message.get('try_times', 0)

        # Execute SQL
        error_info = self._execute_sql(old_sql, sqldb)
        # import ipdb; ipdb.set_trace()
        # Determine whether refinement is needed
        # import ipdb; ipdb.set_trace()
        need_refine = self._is_need_refine(error_info, try_times)

        if not need_refine:
            if " || ' ' || " in old_sql:
                old_sql = old_sql.replace(" || ' ' || ", ", ")
            old_sql = old_sql.replace('ASC LIMIT', 'ASC NULLS LAST LIMIT')
            # import ipdb; ipdb.set_trace()
            # print("Final predicted sql: ", old_sql)
            message['try_times'] = try_times + 1
            message['pred'] = old_sql

            if len(message['subquery_list']) == 1:
                message['send_to'] = SYSTEM_NAME
            else:
                message['last_subquery'] = message['subquery_list'][0]
                message['subquery_list'].pop(0)
                message['sub_sql'] = old_sql
                message['try_times'] = 0
                message.pop('pred', None)
                message['send_to'] = GENERATOR_NAME
        else:
            # Refine the SQL
            # import ipdb; ipdb.set_trace()
            new_sql, filter_error = self._refine(
                sqldb=sqldb,
                query=query,
                evidence=evidence,
                schema_info=schema_info,
                pk_info=pk_info,
                fk_info=fk_info,
                column_details=column_details,
                error_info=error_info,
                complete_schema=complete_schema,
                matched_content=matched_content
            )

            if filter_error:
                message['desc_str'] = message['complete_desc_str']

            message['try_times'] = try_times + 1
            message['pred'] = new_sql
            message['fixed'] = True
            message['send_to'] = REFINER_NAME

if __name__ == "__main__":
    m = 0