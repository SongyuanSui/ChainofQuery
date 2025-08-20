import re
import copy
import numpy as np
import pandas as pd
import random

from utils.myllm import MyChatGPT
from utils.helper import table2string
from utils.normalizer import convert_df_type, prepare_df_for_mysqldb_from_table
from utils.general_prompt import *

sql_two_shot_example_wikitq = '''Given the table schema and three example rows out of the table, write a SQLite program to extract the sub-table that contains the information needed to answer the question.
The SQLite does not need to directly answer the question.
Assume you always have enough information when executing the SQLite.
Output only the SQL, with no explanation.
[Response format] Your response should be in this format:
SQL:
```sql
[the completed SQL]
```

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
SQL:
```sql
WITH Wins AS (
    SELECT
    name,
    CAST(SUBSTR(career_nwin_loss, 1, INSTR(career_nwin_loss, '-') - 1) AS INT) AS wins,
    CAST(SUBSTR(career_nwin_loss, INSTR(career_nwin_loss, '-') + 1) AS INT) AS losses
    FROM Fabrice_Santoro
    WHERE name LIKE "%australian open%" OR name LIKE "%indian wells%"
)
SELECT name, SUM(wins) as total_wins, SUM(losses) as total_losses FROM Wins GROUP BY name;
```

Table:
CREATE TABLE Playa_de_Oro_International_Airport(
	row_id int,
	rank text,
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
SQL:
```sql
WITH PassengerCounts AS (
    SELECT
    city,
    CAST(REPLACE(passengers, ',', '') AS INT) AS passenger_count
    FROM Playa_de_Oro_International_Airport
    WHERE city LIKE "%los angeles%" OR city LIKE "%saskatoon%"
)
SELECT
SUM(CASE WHEN city LIKE "%los angeles%" THEN passenger_count ELSE 0 END) -
SUM(CASE WHEN city LIKE "%saskatoon%" THEN passenger_count ELSE 0 END) AS passenger_difference
FROM PassengerCounts;
```
'''

def base_sql_agent(llm: MyChatGPT, question: str, prompt_schema: str, title: str, num_rows: int = 3,
                        llm_options = None, debug = False, strategy="top") -> str:
    if llm_options is None:
        llm_options = llm.get_model_options()
    prompt = sql_two_shot_example_wikitq + f"\nTable:\n{prompt_schema}Question: {question}\nSQLite:\n"
    if debug:
        print("Final prompt:\n", prompt)
    response = llm.generate(prompt = prompt, options = llm_options)
    sql_query = extract_sql(response)
    return sql_query