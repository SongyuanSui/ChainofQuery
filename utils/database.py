import copy
import os
import sqlite3
import records
import pandas as pd
from typing import Dict, List
import uuid
import re
import threading
from utils.normalizer import convert_df_type, prepare_df_for_mysqldb_from_table
from sqlalchemy.exc import OperationalError, SQLAlchemyError

def fix_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        duplicate_indices = cols[cols == dup].index.values.tolist()
        cols[duplicate_indices] = [dup] + [f"{dup}_{i}" for i in range(1, sum(cols == dup))]
    df.columns = cols
    return df

def check_in_and_return(key: str, source: dict):
    if key.startswith("`") and key.endswith("`"):
        key = key[1:-1]
    if key in source:
        return source[key]
    for _k, _v in source.items():
        if _k.lower() == key.lower():
            return _v
    raise ValueError(f"'{key}' not found in the provided dictionary.")

def make_sqlite_friendly(name: str):
    if not name:
        return "_unnamed"
    name = re.sub(r'[^\w\u4e00-\u9fff]', '_', name)
    return name if not name[0].isdigit() else '_' + name

class MYSQLDB(object):
    def __init__(self, tables: List[Dict[str, Dict]]):
        self.raw_tables = copy.deepcopy(tables)
        for table_info in tables:
            table_info['table'] = prepare_df_for_mysqldb_from_table(
                table_info['table'], normalize=True, add_row_id=False
            )
        self.tables = tables
        self.tmp_path = "tmp"
        os.makedirs(self.tmp_path, exist_ok=True)
        self.db_path = os.path.join(self.tmp_path, '{}.db'.format(uuid.uuid4()))
        self.sqlite_conn = sqlite3.connect(self.db_path)
        assert len(tables) >= 1, "Database must contain at least one table."
        self.table_names, self.table_dict = [], {}
        for table in tables:
            new_column_names = [make_sqlite_friendly(name) for name in table["table"].columns.tolist()]
            table["table"].rename(columns=dict(zip(table["table"].columns.tolist(), new_column_names)), inplace=True)
            table["table"] = fix_duplicate_columns(table["table"])
            table_title = table.get('title', None)
            table_name = make_sqlite_friendly(table_title) if table_title else 'table_name'
            try:
                table["table"].to_sql(table_name, self.sqlite_conn)
            except Exception as e:
                print(f"Error storing table '{table_name}': {e}")
            self.table_names.append(table_name)
            self.table_dict[table_name] = table["table"]
        self.db = records.Database('sqlite:///{}'.format(self.db_path))
        self.records_conn = self.db.get_connection()
        self.creator_thread_id = threading.get_ident()
        self._closed = False

    def __str__(self):
        return str(self.execute_query(f"SELECT * FROM {self.table_names[0]}"))

    def get_table(self, table_name=None):
        table_name = self.table_names[0] if not table_name else table_name
        return self.execute_query(f"SELECT * FROM {table_name}")

    def get_table_schema(self, table_name: str):
        try:
            sql_row_count = f"SELECT COUNT(*) AS count FROM {table_name};"
            count_result = self.records_conn.query(sql_row_count)
            row_count = count_result.first()["count"]

            sql_query = f"PRAGMA table_info({table_name});"
            result = self.records_conn.query(sql_query)
            schema = [(row['name'], row['type'] or 'UNKNOWN') for row in result.all()]
            return row_count, schema
        except Exception as e:
            print(f"[Schema Error] Failed to get schema for '{table_name}': {e}")
            return 0, []

    def get_primary_keys(self, table_name: str):
        try:
            sql_query = f"PRAGMA table_info({table_name});"
            result = self.records_conn.query(sql_query)
            pk_cols = [row['name'] for row in result.all() if row['pk'] == 1]
            return pk_cols
        except Exception as e:
            print(f"[Primary Key Error] Failed to get primary keys for '{table_name}': {e}")
            return []

    def get_header(self, table_name=None):
        return self.get_table(table_name)['header']

    def get_rows(self, table_name):
        return self.get_table(table_name)['rows']

    def get_table_df(self):
        return self.tables[0]['table']

    def get_table_raw(self):
        return self.raw_tables[0]['table']

    def get_table_title(self):
        return self.table_names[0]

    def execute_query(self, sql_query: str) -> dict:
        try:
            result = self.records_conn.query(sql_query)
            headers = result.dataset.headers
            rows = result.all()
            if not headers and rows:
                return {
                    "header": [],
                    "rows": [list(row.values()) for row in rows],
                    "sql": sql_query,
                    "sqlite_error": "Headers missing but rows present, possible metadata anomaly.",
                    "exception_class": "Missing Header Warning"
                }
            if headers and not rows:
                return {
                    "header": headers,
                    "rows": [],
                    "sql": sql_query,
                    "sqlite_error": "SQL query returned zero rows.",
                    "exception_class": "No Row Warning"
                }
            if not headers and not rows:
                return {
                    "header": [],
                    "rows": [],
                    "sql": sql_query,
                    "sqlite_error": "Query returned neither headers nor rows.",
                    "exception_class": "Empty Result Warning"
                }
            return {
                "header": headers,
                "rows": [list(row.values()) for row in rows],
                "sql": sql_query,
                "sqlite_error": "",
                "exception_class": ""
            }
        except OperationalError as oe:
            return {
                "header": [],
                "rows": [],
                "sql": sql_query,
                "sqlite_error": str(oe),
                "exception_class": type(oe).__name__
            }
        except SQLAlchemyError as se:
            return {
                "header": [],
                "rows": [],
                "sql": sql_query,
                "sqlite_error": str(se),
                "exception_class": type(se).__name__
            }
        except Exception as e:
            return {
                "header": [],
                "rows": [],
                "sql": sql_query,
                "sqlite_error": str(e),
                "exception_class": type(e).__name__
            }

    def execute_sql_noreturn(self, sql_statement: str) -> dict:
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute(sql_statement)
            self.sqlite_conn.commit()
            return {
                "sql": sql_statement,
                "success": True,
                "sqlite_error": "",
                "exception_class": ""
            }
        except OperationalError as oe:
            return {
                "sql": sql_statement,
                "success": False,
                "sqlite_error": str(oe),
                "exception_class": type(oe).__name__
            }
        except SQLAlchemyError as se:
            return {
                "sql": sql_statement,
                "success": False,
                "sqlite_error": str(se),
                "exception_class": type(se).__name__
            }
        except Exception as e:
            return {
                "sql": sql_statement,
                "success": False,
                "sqlite_error": str(e),
                "exception_class": type(e).__name__
            }

    def add_sub_table(self, sub_table, table_name=None, verbose=True):
        table_name = self.table_names[0] if not table_name else table_name
        sql_query = f"SELECT * FROM {table_name}"
        old_table_df = pd.DataFrame(self.execute_query(sql_query)["rows"],
                                    columns=self.execute_query(sql_query)["header"])
        sub_table_df = convert_df_type(pd.DataFrame(data=sub_table['rows'], columns=sub_table['header']))
        new_table = old_table_df.merge(sub_table_df, how='left', on='row_id')
        new_table.to_sql(table_name, self.sqlite_conn, if_exists='replace', index=False)
        if verbose:
            print(f"Inserted columns {', '.join(sub_table['header'])} into table '{table_name}'.")

    def close(self):
        if self._closed:
            return
        try:
            if hasattr(self, 'sqlite_conn') and self.sqlite_conn:
                self.sqlite_conn.close()
            if hasattr(self, 'records_conn') and self.records_conn:
                self.records_conn.close()
            if hasattr(self, 'db_path') and os.path.exists(self.db_path):
                os.remove(self.db_path)
        except Exception as e:
            print(f"[MYSQLDB Error] Failed to close: {e}")
        finally:
            self._closed = True

    def __del__(self):
        if not self._closed:
            current_thread_id = threading.get_ident()
            if current_thread_id != self.creator_thread_id:
                print(f"[MYSQLDB Warning] __del__ called from a different thread. Please call `.close()` manually.")
                return
            self.close()