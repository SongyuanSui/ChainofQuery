MAX_ROUND = 8  # MAX_ROUND + 2 is the max try times of one task


SCHEMALINKER_NAME = 'Soft_Schema_Linker'
DECOMPOSER_NAME = 'Decomposer'
GENERATOR_NAME = 'Generator'
REFINER_NAME = 'Refiner'
SYSTEM_NAME = 'System'


schema_linker_template = """
As an experienced and professional database administrator, your task is to analyze a user question and a database schema to provide relevant information. The database schema consists of table descriptions, each containing multiple column descriptions. Your goal is to extract the entities from question and identify the relevant tables and columns based on these entities and the evidence provided.

[Instruction]:
1. Extract the mentioned entities from the user question. Make sure all of the entities are extracted.
2. For each entity, keep at least 3 related columns.
4. Your output should include entity extraction, analysis and related database schema.
5. The related database schema should be in JSON format.
6. Each column's information in provided 【Schema】 is in this format: (column, description. Value examples<optional>)

[Requirements]:
1. Sort the related columns in each list corresponding to each entity in descending order of relevance.
2. The chosen columns should be in this format: <table.column>.
3. Make sure each chosen list is not empty. The value [] will be punished.
4.【Matched values】 may contain redundant or useless information in addition to the correct matching values, so you need to select the useful information in conjunction with the specific column names and descriptions.
5. An entity may not have a corresponding evidence, which requires you to find the relevant columns yourself through your understanding of the database schema.
6. There are columns with similar names but different meanings, you can find the most accurate columns in the right table based on the 【Summary of each table】.

Here is a typical example:

==========
【DB_ID】 banking_system
【Schema】
# Table: account
[
  (account_id <INTEGER>, the id of the account. Value examples: [11382, 11362, 2, 1, 2367].),
  (district_id <INTEGER>, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (frequency <TEXT>, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'].),
  (date <DATE>, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)
]
# Table: client
[
  (client_id <INTEGER>, the unique number. Value examples: [13998, 13971, 2, 1, 2839].),
  (gender <TEXT>, gender. Value examples: ['M', 'F']. And F：female . M：male ),
  (birth_date <DATE>, birth date. Value examples: ['1987-09-27', '1986-08-13'].),
  (district_id <INTEGER>, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (first_name <TEXT>, first_name.),
  (last_name <TEXT>, last_name.)
]
# Table: loan
[
  (loan_id <INTEGER>, the id number identifying the loan data. Value examples: [4959, 4960, 4961].),
  (account_id <INTEGER>, the id number identifying the account. Value examples: [10, 80, 55, 43].),
  (date <DATE>, the date when the loan is approved. Value examples: ['1998-07-12', '1998-04-19'].),
  (amount <INTEGER>, the id number identifying the loan data. Value examples: [1567, 7877, 9988].),
  (duration <INTEGER>, the id number identifying the loan data. Value examples: [60, 48, 24, 12, 36].),
  (payments <INTEGER>, the id number identifying the loan data. Value examples: [3456, 8972, 9845].),
  (status <TEXT>, the id number identifying the loan data. Value examples: ['C', 'A', 'D', 'B'].)
]
# Table: district
[
  (district_id <INTEGER>, location of branch. Value examples: [77, 76].),
  (A2 <REAL>, area in square kilometers. Value examples: [50.5, 48.9].),
  (A4 <INTEGER>, number of inhabitants. Value examples: [95907, 95616].),
  (A5 <INTEGER>, number of households. Value examples: [35678, 34892].),
  (A6 <REAL>, literacy rate. Value examples: [95.6, 92.3, 89.7].),
  (A7 <INTEGER>, number of entrepreneurs. Value examples: [1234, 1456].),
  (A8 <INTEGERt>, number of cities. Value examples: [5, 4].),
  (A9 <INTEGER>, number of schools. Value examples: [15, 12, 10].),
  (A10 <INTEGER>, number of hospitals. Value examples: [8, 6, 4].),
  (A11 <REAL>, average salary. Value examples: [12541.5, 11277].),
  (A12 <REAL>, poverty rate. Value examples: [12.4, 9.8].),
  (A13 <REAL>, unemployment rate. Value examples: [8.2, 7.9].),
  (A15 <INTEGER>, number of crimes. Value examples: [256, 189].)
]
【Primary keys】
account.`account_id` | client.`client_id` | loan.`loan_id` | district.`district_id`
【Foreign keys】
client.`district_id` = district.`district_id`
【Question】
What is the gender of the youngest client who opened account in the lowest average salary branch and when did this client open the account? Please list their full name.
【Evidence】
Later birthdate refers to younger age; A11 refers to average salary; Full name refers to first_name, last_name
【Matched values】
Since some of the specific values in Question and evidence match the data in the database, here are some matches retrieved from the database that may help you in selecting columns (You need to ignore matches that are not relevant to the question):
No matched values.
【Summary of each table】
# account: Specific information for each account
# client: Basic information about each client
# loan: Detailed records of each loan
# district: Various data recorded in each district
【Answer】
The entities extracted from the 【Question】 are:
1. gender;
2. youngest client;
3. account;
4. lowest average salary branch;
5. when did this client open the account;
6. full name

Extract the related evidence in 【Evidence】 for entities if 【Evidence】 is not None:
1. gender --no related evidence **the columns about gender are related**
2. youngest client  --Later birthdate refers to younger age; **the columns about birth date of client and the id of client are related**
3. account --no related evidence **the columns about account or account ids are related**
4. lowest average salary branch  --A11 refers to average salary **the columns about average salary and branch are related**
5. when did this client open the account --no related evidence **the columns about time or date of the account are related**
6. full name  --Full name refers to first_name, last_name **the columns about first_name, last_name of the client are related**

Therefore, we can select the related database schema based on these entities with 【Evidence】:
```json
{{
  "gender": ["client.gender","client.client_id","loan.status"],
  "youngest client": ["client.birth_date","client.client_id","account.date","loan.date"],
  "account": ["account.account_id","loan.account_id","account.date"],
  "lowest average salary branch": ["district.A11","district.district_id","district.A13"],
  "when did this client open the account": ["account.date","loan.date","client.birth_date"],
  "full name": ["client.first_name","client.last_name","client.client_id"]
}}
```
Question Solved.

==========

Here is a new example, please start answering:

【DB_ID】 {db_id}
【Schema】
{desc_str}
【Primary keys】
{pk_str}
【Foreign keys】
{fk_str}
【Question】
{query}
【Evidence】
{evidence}
【Matched values】
Since some of the specific values in Question and evidence match the data in the database, here are some matches retrieved from the database that may help you in selecting columns (You need to ignore matches that are not relevant to the question):
{matched_list}
【Summary of each table】
{summary_str}
【Answer】
"""

summarizer_template = """
[instruction]
Given the database schema, you need to summarise the data stored in each table in one sentence, based on the name of the table and the columns in the table.

[Requirements]
- Your output should be in json format

Here is an example:
==========

【DB_ID】 banking_system
【Schema】
# Table: account
[
  (account_id <INTEGER>, the id of the account. Value examples: [11382, 11362, 2, 1, 2367].),
  (district_id <INTEGER>, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (frequency <TEXT>, frequency of the acount. Value examples: ['POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'].),
  (date <DATE>, the creation date of the account. Value examples: ['1997-12-29', '1997-12-28'].)
]
# Table: client
[
  (client_id <INTEGER>, the unique number. Value examples: [13998, 13971, 2, 1, 2839].),
  (gender <TEXT>, gender. Value examples: ['M', 'F']. And F：female . M：male ),
  (birth_date <DATE>, birth date. Value examples: ['1987-09-27', '1986-08-13'].),
  (district_id <INTEGER>, location of branch. Value examples: [77, 76, 2, 1, 39].),
  (first_name <TEXT>, first_name.),
  (last_name <TEXT>, last_name.)
]
# Table: loan
[
  (loan_id <INTEGER>, the id number identifying the loan data. Value examples: [4959, 4960, 4961].),
  (account_id <INTEGER>, the id number identifying the account. Value examples: [10, 80, 55, 43].),
  (date <DATE>, the date when the loan is approved. Value examples: ['1998-07-12', '1998-04-19'].),
  (amount <INTEGER>, the id number identifying the loan data. Value examples: [1567, 7877, 9988].),
  (duration <INTEGER>, the id number identifying the loan data. Value examples: [60, 48, 24, 12, 36].),
  (payments <INTEGER>, the id number identifying the loan data. Value examples: [3456, 8972, 9845].),
  (status <TEXT>, the id number identifying the loan data. Value examples: ['C', 'A', 'D', 'B'].)
]
# Table: district
[
  (district_id <INTEGER>, location of branch. Value examples: [77, 76].),
  (A2 <REAL>, area in square kilometers. Value examples: [50.5, 48.9].),
  (A4 <INTEGER>, number of inhabitants. Value examples: [95907, 95616].),
  (A5 <INTEGER>, number of households. Value examples: [35678, 34892].),
  (A6 <REAL>, literacy rate. Value examples: [95.6, 92.3, 89.7].),
  (A7 <INTEGER>, number of entrepreneurs. Value examples: [1234, 1456].),
  (A8 <INTEGERt>, number of cities. Value examples: [5, 4].),
  (A9 <INTEGER>, number of schools. Value examples: [15, 12, 10].),
  (A10 <INTEGER>, number of hospitals. Value examples: [8, 6, 4].),
  (A11 <REAL>, average salary. Value examples: [12541.5, 11277].),
  (A12 <REAL>, poverty rate. Value examples: [12.4, 9.8].),
  (A13 <REAL>, unemployment rate. Value examples: [8.2, 7.9].),
  (A15 <INTEGER>, number of crimes. Value examples: [256, 189].)
]

【Summary】
```json
{{
    "account":"Specific information for each account",
    "client":"Basic information about each client",
    "loan":"Detailed records of each loan",
    "district":"Various data recorded in each district",

}}
```
==========

Here is a new case:
【DB_ID】 {db_id}
【Schema】
{desc_str}

【Summary】
"""


subq_pattern = r"Sub question\s*\d+\s*:"


pure_decomposer_template = """
[Instruction]
Given a 【query】, you need to understanding the intent of Query, and then deceompose it into Targets and Conditions. Then you need to combine Targets and Conditions into Subquerys step by step.
For the case where Conditions is NULL, consider Targets as the final Subquery directly.
For the case where Conditions are not NULL, combine Targets and the first Condition to get the first Subquery, then combine this Subquery and the next Condition into a new Subquery until all Conditions are used (which means the content of the last Subquery and the original Query is the same).

[Requirements]
-Try not to overlap Targets and Conditions.
-Make sure the decomposed Target and Condition can cover all of the information in Query.
-Don't change any information (specific value) in Query!
-Mark each Subquery with ## in front of it.

Here are some examples:
==========

【Query】
Show the stadium name and the number of concerts in each stadium. Please also list the year the stadium was built.
【Evidence】
NULL

【Decomposition】
Targets: List the stadium name, the year the stadium built and the number of concerts in each stadium
Conditions: NULL

Subqueries:
1. Since Conditions is NULL, the final Subquery is the Targets.
##Subquery: List the stadium name, the year the stadium built and the number of concerts in each stadium
==========

【Query】
What is the qualification rate for the H-11 products produced in 2023/11/2?
【Evidence】
qualification rate = `Numqualified(H-11)` / `production(H-11)`

【Decomposition】
Targets: List the qualification rate for the H-11 Products
Conditions:
1. produced in 2023/11/2 --Condition_1

Subqueries:
1. Combine Targets and Conditon_1 to get the first Subquery.
##Subquery: List the qualification rate for the H-11 Products produced in 2023/11/2

==========

【Query】
List the race of institutions in Alabama with number of students greater than the 90% of average number of students of all institutions?
【Evidence】
Alabama refers to state = 'Alabama'; number of students greater than the 90% of average = MULTIPLY(AVG(student_count), 90%) < student_count

【Decomposition】
Targets: List the race of institutions
Conditions:
1. in Alabama --Condition_1
2. number of students greater than the 90% of average number of students of all institutions --Condition_2

Subqueries:
1. Combine Targets and Condition_1 to get the first Subquery.
##Subquery: List the race of institution in Alabama
2. Conbine the first Subquery and Conditon_2 to get the seconed Subquery.
##Subquery: List the race of institutions in Alabama with number of number of students greater than the 90% of average number of students of all institutions

==========

【Query】
Which president got a higher approval rating, Joseph Biden or Donald Trump?
【Evidence】
NULL

【Decomposition】
Targets: List the name of the president who got a higher approval rating between Joseph Biden or Donald Trump
Conditions:
NULL

Subqueries:
1. Since Conditions is NULL, the final Subquery is the Targets.
##Subquery: List the name of the president who got a higher approval rating between Joseph Biden or Donald Trump

==========

【Query】
For movie id 1269, how many users, who was a paying subscriber and was eligible for trial when he rated the movie, gave the movie a rating score of less than or equal to 2?
【Evidence】
NULL

【Decomposition】
Targets: List the number of users
Conditions:
1. was a paying subscriber --Condition_1
2. was eligible for trial --Condition_2
3. for movie id 1269, gave the movie a rating score of less than or equal to 2 --Condition3

Subquerys:
1. Combine Targets and Conditon_1 to get the first Subquery.
##Subquery: List the number of users who was a paying subscriber
2. Combine the first Subquery and Condition_2 to get the second Subquery.
##Subquery: List the number of users who was a paying subscriber and was eligible for trial
3. Combine the second Subquery and Condition_3 to get the third Subquery.
##Subquery: List the number of users who was a paying subscriber and was eligible for trial and gave the movie whose id is 1269 a rating score of less than or equal to 2

==========

Here is a new query need to be decomposed:

【Query】
{query}
【Evidence】
{evidence}

【Decomposition】
"""

soft_schema_initial_generator_template = """
Given a 【Database schema】 description, a knowledge 【Evidence】and a 【Question】, you need to use valid SQLite and understand the database and knowledge so that you can generate a good SQL for the 【Question】.
When generating SQL, we should always consider constraints:
【Constraints】
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values
- If include more than one table, use `JOIN <table>`
- If use `JOIN <table>`, the connected columns should be in the 【Foreign keys】
- If evidence gives a formula for calculating a value, try to use that formula
- If use `ORDER BY <column> ASC LIMIT <n>`, please use `ORDER BY <column> ASC NULLS LAST LIMIT <n>` to make sure the null values will not be selected

==========

【Database schema】
# stadium: [Stadium_ID (INTEGER), Location (TEXT), Name (TEXT)]
# concert: [concert_ID (INTEGER), concert_Name (TEXT), Stadium_ID (INTEGER)]
【Primary keys】
stadium.`Stadium_ID` | concert.`concert_ID`
【Foreign keys】
concert.`Stadium_ID` = stadium.`Stadium_ID`
【Detailed descriptions of tables and columns】
stadium.`Stadium_ID`: The column 'Stadium_ID' in Table <stadium> has column descriptions of "stadium id". Value examples: [1, 2, 3, 4, 5, 6].
stadium.`Name`:  The column 'Name' in Table <stadium> has column descriptions of "name of stadium". Value examples: ["Stark's Park", 'Somerset Park', 'Recreation Park', 'Hampden Park', 'Glebe Park', 'Gayfield Park'].
stadium.`Location`: The column 'Location' in Table <stadium> has column descriptions of "location of stadium". Value examples: ['Stirling Albion', 'Raith Rovers', "Queen's Park", 'Peterhead', 'East Fife', 'Brechin City'].
concert.`concert_ID`: The column 'concert_ID' in Table <concert> has column descriptions of "concert id". Value examples: [1, 2, 3, 4, 5, 6].
concert.`concert_Name`: The column 'concert_Name' in Table <concert> has column descriptions of "concert name". Value examples: ['Week 1', 'Week 2', 'Super bootcamp', 'Home Visits', 'Auditions'].
concert.`Stadium_ID`: The column 'Stadium_ID' in Table <concert> has column descriptions of "stadium id". Value examples: [2, 9, 7, 10, 1].
【Evidence】
NULL
【Question】
Show the stadium name and the number of concerts in each stadium.
【Matched values】
Since some of the specific values in Question and evidence match the data in the database, here are some matches retrieved from the database that may help you to generate SQL (Matched values may contain useless information and you should ignore matches that are not relevant to the question):
No matched values.

Consider 【Constraints】, extract hints from 【Evidence】 if 【Evidence】 is related to the 【Question】, select columns from 【Database schema】 and then generate SQL for 【Question】, you need to think step by step:
【Question】: Show the stadium name and the number of concerts in each stadium.
Targets for `SELECT`: the stadium name and the number of concerts
hints from 【Evidence】:NULL
For the entities in the 【Question】, get corresponding columns from 【Database schema】 with hints: the stadium name refers to stadium.`Name`, the number of concerts refers to COUNT(concert.`concert_ID`), each stadium refers to stadium.`Stadium_ID`
Connection of tables: include tables <stadium> and <concert>, get connected keys from 【Foreign keys】: concert.`Stadium_ID` = stadium.`Stadium_ID`
Final SQL:
```sql
SELECT T1.`Name`, COUNT(T2.concert_ID) AS num_concerts
  FROM stadium AS T1
  JOIN concert AS T2
  ON T1.`Stadium_ID` = T2.`Stadium_ID`
  GROUP BY T1.`Stadium_ID`
```

Question Solved.

==========

【Database schema】
# country: [origin (INTEGER), country(TEXT)]
# price: [ID (INTEGER), price (REAL)]
# data: [ID (INTEGER), mpg (REAL), cylinders (INTEGER), displacement (TEXT), horsepower (REAL), weight (REAL), acceleration (REAL), model (TEXT), car_name (TEXT)]
# production: [ID (INTEGER), model_year (INTEGER), country (INTEGER)]
【Primary keys】
country.`origin` | price.`ID` | data.`ID` | production.`ID`
【Foreign keys】
data.`ID` = price.`ID`
production.`ID` = price.`ID`
production.`ID` = data.`ID`
production.`country` = country.`origin`
【Detailed descriptions of tables and columns】
country.`origin`: The column 'origin' in Table <country> has column descriptions of "the unique identifier for the origin country". Value examples: [1, 2, 3].
country.`country`: The column 'country' in Table <country> has column descriptions of "the origin country of the car". Value examples: ['USA', 'Japan', 'Europe'].
data.`horsepower`:  The column 'horsepower' in Table <data> has column descriptions of "horse power associated with the car".
data.`acceleration`:  The column 'acceleration' in Table <data> has column descriptions of "acceleration of the car in miles per squared hour".
production.`country`: The column 'country' in Table <production> has column descriptions of "country id to which the car belongs".  Value examples: [1, 2, 3].
【Evidence】
the fastest refers to max(horsepower); name of the car refers to car_name
【Question】
What is the fastest car made by Japan?
【Matched values】
Since some of the specific values in Question and evidence match the data in the database, here are some matches retrieved from the database that may help you to generate SQL (Matched values may contain useless information and you should ignore matches that are not relevant to the question):
country.`country` = 'Japan'

Consider 【Constraints】, extract hints from 【Evidence】 if 【Evidence】 is related to the 【Question】, select columns from 【Database schema】 and then generate SQL for 【Question】, you need to think step by step:
【Question】: What is the fastest car made by Japan?
Targets for `SELECT`: the name of the fastest car
hints from 【Evidence】: the fastest refers to max(horsepower); name of the car refers to car_name
For the entities in the 【Question】, get corresponding columns from 【Database schema】 with hints: car refers to data.`car_name`, the fastest refers to MAX(data.`horsepower`), Japan refers to country.`country`
Connection of tables: includes tables <data> and <country>, since <data> and <country> are not connected directly, use <production> as a bridge, get connected keys from 【Foreign keys】: production.`ID` = data.`ID`, production.`country` = country.`origin`
Final SQL:
```sql
SELECT T1.`car_name`
  FROM data AS T1
  INNER JOIN production AS T2
  ON T1.`ID` = T2.`ID`
  INNER JOIN country AS T3
  ON T3.`origin` = T2.`country`
  WHERE T3.`country` = 'Japan'
  ORDER BY T1.`horsepower` DESC LIMIT 1
```

Question Solved.

==========

【Database schema】
# institution_details: [unitid (INTEGER), chronname (TEXT), city (TEXT), state (TEXT), site (TEXT), student_count(TEXT)]
# institution_grads: [unitid (INTEGER), gender (TEXT), race (TEXT), cohhort (TEXT)]
【Primary keys】
institution_details.`unitid` | institution_grads.`unitid`
【Foreign keys】
institution_grads.`unitid` = institution_details.`unitid`
【Detailed descriptions of tables and columns】
institution_details.`unitid`: The column 'unitid' in Table <institution_details> has column descriptions of "Education Unit ID number".
institution_details.`chronname`: The column 'chronname' in Table <institution_details> has column descriptions of "Institution name".
institution_details.`student_count`: The column 'student_count' in Table <institution_details> has column descriptions of "Total number of undergraduates in 2010".
【Evidence】
number of students greater than the 90% of average = MULTIPLY(AVG(student_count), 90%) < student_count
【Question】
List the chronname of institutions with number of students greater than the 90% of average number of students of all institutions?
【Matched values】
Since some of the specific values in Question and evidence match the data in the database, here are some matches retrieved from the database that may help you to generate SQL (Matched values may contain useless information and you should ignore matches that are not relevant to the question):
No matched values.

Consider 【Constraints】, extract hints from 【Evidence】 if 【Evidence】 is related to the 【Question】, select columns from 【Database schema】 and then generate SQL for 【Question】, you need to think step by step:
【Question】: List the chroname of institutions with number of students greater than the 90% of average number of students of all institutions?
Targets for `SELECT`: the chroname of institutions
hints from 【Evidence】: number of students greater than the 90% of average = MULTIPLY(AVG(student_count), 90%) < student_count
There is a formula in 【Evidence】, so we can turn it into SQL format: institution_details.`student_count` > ( SELECT AVG(`student_count`) * 0.9 FROM institution_details )
For the entities in the 【Question】, get corresponding columns from 【Database schema】 with hints: the chronname of institutions refers to institution_details.`chronname`, number of students refers to institution_details.`student_count`
Connection of tables: include only one table <institution_details>, no connection
Final SQL:
```sql
SELECT DISTINCT `chroname`
  FROM institution_details
  WHERE `student_count` > (
    SELECT AVG(`student_count`) * 0.9 FROM institution_details
  )
```

Question Solved.

==========

【Database schema】
{desc_str}
【Primary keys】
{pk_str}
【Foreign keys】
{fk_str}
【Detailed descriptions of tables and columns】
{detailed_str}
【Evidence】
{evidence}
【Question】
{query}
【Matched values】
Since some of the specific values in Question and evidence match the data in the database, here are some matches retrieved from the database that may help you to generate SQL (Matched values may contain useless information and you should ignore matches that are not relevant to the question):
{matched_list}

Consider 【Constraints】, extract hints from 【Evidence】 if 【Evidence】 is related to the 【Question】, select columns from 【Database schema】 and then generate SQL for 【Question】, you need to think step by step:
"""

soft_schema_continuous_generator_template = """
Given a 【Database schema】 description, a knowledge 【Evidence】, a 【Question】, a 【Subquesion】 extracted from 【Question】 and a 【Sub-SQL】 for the 【Subquestion】, you need to use valid SQLite and understand the database and knowledge, and then generate a complete SQL for the 【Question】 based on 【Sub-SQL】.
When generating SQL, we should always consider constraints:
【Constraints】
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use max or min func, `JOIN <table>` FIRST, THEN use `SELECT MAX(<column>)` or `SELECT MIN(<column>)`
- If [Value examples] of <column> has 'None' or None, use `JOIN <table>` or `WHERE <column> is NOT NULL` is better
- If use `ORDER BY <column> ASC|DESC`, add `GROUP BY <column>` before to select distinct values
- If include more than one table, use `JOIN <table>`
- If use `JOIN <table>`, the connected columns should be in the 【Foreign keys】
- If evidence gives a formula for calculating a value, try to use that formula
- If use `ORDER BY <column> ASC LIMIT <n>`, please use `ORDER BY <column> ASC NULLS LAST LIMIT <n>` to make sure the null values will not be selected

==========

【Database schema】
# country: [origin (INTEGER), country(TEXT)]
# price: [ID (INTEGER), price (REAL)]
# data: [ID (INTEGER), mpg (REAL), cylinders (INTEGER), displacement (TEXT), horsepower (REAL), weight (REAL), acceleration (REAL), model (TEXT), car_name (TEXT)]
# production: [ID (INTEGER), model_year (INTEGER), country (INTEGER)]
【Primary keys】
country.`origin` | price.`ID` | data.`ID` | production.`ID`
【Foreign keys】
data.`ID` = price.`ID`
production.`ID` = price.`ID`
production.`ID` = data.`ID`
production.`country` = country.`origin`
【Detailed descriptions of tables and columns】
country.`origin`: The column 'origin' in Table <country> has column descriptions of "the unique identifier for the origin country". Value examples: [1, 2, 3].
country.`country`: The column 'country' in Table <country> has column descriptions of "the origin country of the car". Value examples: ['USA', 'Japan', 'Europe'].
data.`horsepower`:  The column 'horsepower' in Table <data> has column descriptions of "horse power associated with the car".
data.`acceleration`:  The column 'acceleration' in Table <data> has column descriptions of "acceleration of the car in miles per squared hour".
production.`country`: The column 'country' in Table <production> has column descriptions of "country id to which the car belongs".  Value examples: [1, 2, 3].
【Evidence】
the fastest refers to max(horsepower); made by Japan refers to country = 'Japan'
【Question】
What is the price of the fastest car made by Japan?
【Subquestion】
What is the price of the fastest car?
【Sub-SQL】
```
SELECT T1.`price` FROM price AS T1 INNER JOIN data AS T2 ON T2.`ID` = T1.`ID` ORDER BY T2.`horsepower` DESC LIMIT 1
```
【Matched values】
Since some of the specific values in Question and evidence match the data in the database, here are some matches retrieved from the database that may help you to generate SQL (Matched values may contain useless information and you should ignore matches that are not relevant to the question):
No matched values.

【Subquestion】 is decomposed from 【Question】, now we provide Sub-SQL corresponding to Subquestion, you just need to complete the remaining conditions based on Sub-SQL to generate SQL for 【Question】. Consider 【Constraints】, extract hints from 【Evidence】 if 【Evidence】 is related to the 【Question】, and then generate SQL after thinking step by step:
hints from 【Evidence】: the fastest refers to max(horsepower); made by Japan refers to country = 'Japan';
Find necessary columns from 【Database schema】 in addition to Sub-SQL: Japan refers to country.`country`
Connection of tables: the connection of <data> and <country> is needed, since <data> and <country> are not connected directly, use <production> as a bridge, get connected keys from 【Foreign keys】: production.`ID` = data.`ID`, production.`country` = country.`origin`
Final SQL:
```sql
SELECT T1.`price`
  FROM price AS T1
  INNER JOIN data AS T2
  on T2.`ID` = T1.`ID`
  INNER JOIN production AS T3
  ON T3.`ID` = T2.`ID`
  INNER JOIN country AS T4
  ON T4.`origin` = T3.`country`
  WHERE T4.`country` = 'Japan'
  ORDER BY T2.`horsepower` DESC LIMIT 1
```

Question solved.

==========

【Database schema】
# institution_details: [unitid (INTEGER), chronname (TEXT), city (TEXT), state (TEXT), site (TEXT), student_count(TEXT)]
# institution_grads: [unitid (INTEGER), gender (TEXT), race (TEXT), cohhort (TEXT)]
【Primary keys】
institution_details.`unitid` | institution_grads.`unitid`
【Foreign keys】
institution_grads.`unitid` = institution_details.`unitid`
【Detailed descriptions of tables and columns】
institution_details.`unitid`: The column 'unitid' in Table <institution_details> has column descriptions of "Education Unit ID number".
institution_details.`state`: The column 'state' in Table <institution_details> has column descriptions of "Institution state".
institution_details.`student_count`: The column 'student_count' in Table <institution_details> has column descriptions of "Total number of undergraduates in 2010".
institution_grads.`race`: The column 'race' in Table <institution_grads> has column descriptions of "race/ethnicity of students". Value examples: ['X', 'Ai', 'A', 'B', 'H'].
【Evidence】
Alabama refers to state = 'Alabama'; number of students greater than the 90% of average = MULTIPLY(AVG(student_count), 90%) < student_count
【Question】
List the race of institutions in Alabama with number of students greater than the 90% of average number of students of all institutions?
【Subquestion】
List the race of institutions in 'Alabama'.
【Sub-SQL】
```
SELECT DISTINCT T1.`race` FROM institution_grads AS T1 INNER JOIN institution_details AS T2 ON T1.`unitid` = T2.`unitid` WHERE T2.`state` = 'Alabama'
```
【Matched values】
Since some of the specific values in Question and evidence match the data in the database, here are some matches retrieved from the database that may help you to generate SQL (Matched values may contain useless information and you should ignore matches that are not relevant to the question):
institution_details.`city` = 'Alabama'; institution_details.`state` = 'Alabama'

【Subquestion】 is decomposed from 【Question】, now we provide Sub-SQL corresponding to Subquestion, you just need to complete the remaining conditions based on Sub-SQL to generate SQL for 【Question】. Consider 【Constraints】, extract hints from 【Evidence】 if 【Evidence】 is related to the 【Question】, and then generate SQL after thinking step by step:
hints from 【Evidence】: Alabama refers to state = 'Alabama'; number of students greater than the 90% of average = MULTIPLY(AVG(student_count), 90%) < student_count
There is a formula in 【Evidence】, so we can turn it into SQL format: institution_details.`student_count` > ( SELECT AVG(`student_count`) * 0.9 FROM institution_details )
Find necessary columns from 【Database schema】 in addition to Sub-SQL: number of students refers to institution_details.`student_count`
Connection of tables: the table <institution_details> is already in Sub-SQL, no more connection is needed
Final SQL:
```sql
SELECT DISTINCT T1.`race`
  FROM institution_grads AS T1
  INNER JOIN institution_details AS T2
  ON T1.`unitid` = T2.`unitid`
  WHERE T2.`student_count` > (
    SELECT AVG(`student_count`) * 0.9 FROM institution_details
  )
  AND T2.`state` = 'Alabama'
```

Question solved.

==========

【Database schema】
{desc_str}
【Primary keys】
{pk_str}
【Foreign keys】
{fk_str}
【Detailed descriptions of tables and columns】
{detailed_str}
【Evidence】
{evidence}
【Question】
{query}
【Subquestion】
{subquery}
【Sub-SQL】
{subsql}
【Matched values】
Since some of the specific values in Question and evidence match the data in the database, here are some matches retrieved from the database that may help you to generate SQL (Matched values may contain useless information and you should ignore matches that are not relevant to the question):
{matched_list}

【Subquestion】 is decomposed from 【Question】, now we provide Sub-SQL corresponding to Subquestion, you just need to complete the remaining conditions based on Sub-SQL to generate SQL for 【Question】. Consider 【Constraints】, extract hints from 【Evidence】 if 【Evidence】 is related to the 【Question】, and then generate SQL after thinking step by step:
"""



baseline_template = """
Given a 【Database schema】 description, a knowledge 【Evidence】 and the 【Question】, you need to use valid SQLite and understand the database and knowledge, and then generate SQL.
You can write answer in script blocks, and indicate script type in it, like this:
```sql
SELECT column_a
FROM table_b
```

【Database schema】
{desc_str}
【Question】
{query}
【Evidence】
{evidence}
【Answer】
"""


refiner_template = """
【Instruction】
When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
Solve the task step by step if you need to. Using SQL format in the code block, and indicate script type in the code block.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
【Constraints】
- The SQL should start with 'SELECT'
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use `JOIN <table>`, the connected columns should be in the Foreign keys of 【Database schema】
【Response format】
Your response should be in this format:
Analysis:
**[Your analysis]**
Correct SQL:
```sql
[the fixed SQL]
```
【Attention】
Only SQL statements are allowed in [the fixed SQL], do not add any comments.

【Evidence】
{evidence}
【Query】
-- {query}
【Database info】
{desc_str}
【Primary keys】
{pk_str}
【Foreign keys】
{fk_str}
【Detailed descriptions of tables and columns]】
{detailed_str}
【old SQL】
```sql
{sql}
```
【SQLite error】
{sqlite_error}
【Exception class】
{exception_class}

Now please fixup old SQL and generate new SQL again.
【correct SQL】
"""

nested_refiner_template = """
【Instruction】
When executing SQL below, some errors occurred, please fix up SQL based on query and database info.
Solve the task step by step if you need to. Using SQL format in the code block, and indicate script type in the code block.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
【Possible error】
When using nested SQL, using the MIN or MAX function in the sub-SQL may result in null data in the end. Because when multiple tables are joined, they are connected by foreign keys and contain only some data in common, the maximum or minimum value of a column in a table may not be in the joined table.
【Constraints】
- The SQL should start with 'SELECT'
- In `SELECT <column>`, just select needed columns in the 【Question】 without any unnecessary column or value
- In `FROM <table>` or `JOIN <table>`, do not include unnecessary table
- If use `JOIN <table>`, the connected columns should be in the Foreign keys of 【Database schema】
- If use `ORDER BY <column> ASC LIMIT <n>`, please use `ORDER BY <column> ASC NULLS LAST LIMIT <n>` to make sure the null values will not be selected
- There are columns with similar names but different meanings, you can find the most accurate columns in the right table based on the 【Summary of each table】
【Response format】
Your response should be in this format:
Analysis:
**[Your analysis]**
Correct SQL:
```sql
[the fixed SQL]
```
【Attention】
Only SQL statements are allowed in [the fixed SQL], do not add any comments.
【Typical examples】
For the error called "no data selected", here are some typical example.
1. Wrong SQL:
```
SELECT T1.`date` FROM recordtime AS T1 INNER JOIN information AS T2 ON T1.`id` = T2.`id` WHERE T2.`viewers` = ( SELECT MAX(`viewers`) FROM information)
```
Analysis: **Using the MAX function in a nested SQL statement will result in a mismatch because the maximum or minimum value of the column may not be in the joined table, so use `ORDER BY`!**
Correct SQL:
```sql
SELECT T1.`date`
  FROM recordtime AS T1
  INNER JOIN information AS T2
  ON T1.`id` = T2.`id`
  ORDER BY T2.`viewers` DESC
  LIMIT 1
```
2. Wrong SQL:
```
SELECT T1.`idea` FROM work AS T1 INNER JOIN student AS T2 ON T1.`ID` = T2.`ID` WHERE T2.`final_score` = ( SELECT MIN(`final_score`) FROM student)
```
Analysis: **Using the MIN function in a nested SQL statement will result in a mismatch because the maximum or minimum value of the column may not be in the joined table, so use `ORDER BY`!**
Correct SQL:
```sql
SELECT T1.`idea`
  FROM work AS T1
  INNER JOIN student AS T2
  ON T1.`ID` = T2.`ID`
  ORDER BY T2.`final_score` ASC
  LIMIT 1
```

Here is a new case for you.
【Evidence】
{evidence}
【Query】
-- {query}
【Database info】
{desc_str}
【Primary keys】
{pk_str}
【Foreign keys】
{fk_str}
【Detailed descriptions of tables and columns]】
{detailed_str}
【Matched values】
{matched_content}
【old SQL】
```sql
{sql}
```
【SQLite error】
{sqlite_error}
【Exception class】
{exception_class}

Now please fixup old SQL and generate new SQL again.
【correct SQL】
"""
