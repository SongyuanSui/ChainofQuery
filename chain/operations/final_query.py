# Copyright 2024 The Chain-of-Table authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import numpy as np
from chain.utils.helper import table2string
from chain.utils.const import DATASET

query_wiki = """/*
col : Rank | City | Passengers Number | Ranking | Airline
row 1 : 1 | United States, Los Angeles | 14749 | 2 | Alaska Airlines
row 2 : 2 | United States, Houston | 5465 | 8 | United Express
row 3 : 3 | Canada, Calgary | 3761 | 5 | Air Transat, WestJet
row 4 : 4 | Canada, Saskatoon | 2282 | 4 |
row 5 : 5 | Canada, Vancouver | 2103 | 2 | Air Transat
row 6 : 6 | United States, Phoenix | 1829 | 1 | US Airways
row 7 : 7 | Canada, Toronto | 1202 | 1 | Air Transat, CanJet
row 8 : 8 | Canada, Edmonton | 110 | 2 |
row 9 : 9 | United States, Oakland | 107 | 5 |
*/
Question: how many more passengers flew to los angeles than to saskatoon from manzanillo airport in 2013?
The anwser is: 12467"""

query_fetaqa = """/*
Table page title: Estadio Elías Aguirre
Table section title: International matches hosted
col : Date | Team #1 | Res. | Team #2 | Round | Attendance | Tournament
row 1 : 7 July 2004 | Mexico | 2–2 | Uruguay | Group B | 25.000 | Copa América
row 2 : 7 July 2004 | Argentina | 6–1 | Ecuador | Group B | 24.000 | Copa América
row 3 : 10 July 2004 | Uruguay | 2–1 | Ecuador | Group B | 25.000 | Copa América
row 4 : 10 July 2004 | Argentina | 0–1 | Mexico | Group B | 25.000 | Copa América
row 5 : 17 July 2004 | Peru | 0–1 | Argentina | Quarterfinals | 26.500 | Copa América
*/
Question: What were the first and final Copa América matches played at Elías Aguirre?
Please provide the answer in a fluent sentence: The game that opened the Elías Aguirre's participation in this tournament was a group stage 2–2 draw between Mexico and Uruguay, and the game that closed was a 1–0 Argentina win over hosts Peru in the quarterfinals.

/*
Table page title: Chevrolet C/K
Table section title: Engines
col : Year | Engine | Power | Torque | Notes
row 1 : 1981–1984 | 4.1 L GMC 250 I-6 | 115 hp (86 kW) @ 3600 RPM | 200 lb⋅ft (271 N⋅m) @ 2000 RPM | -
row 2 : 1983 | 4.1 L GMC 250 I-6 | 120 hp (89 kW) @ 4000 RPM | 205 lb⋅ft (278 N⋅m) @ 2000 RPM | C1
row 3 : 1985–1986 | 4.3 L LB1 90° V-6 | 155 hp (116 kW) @ 4000 RPM | 230 lb⋅ft (312 N⋅m) @ 2400 RPM | -
row 4 : 1987 | 4.3 L LB1 90° V-6 | 160 hp (119 kW) @ 4000 RPM | 235 lb⋅ft (319 N⋅m) @ 2400 RPM | -
row 5 : 1981–1985 | 4.8 L GMC 292 I-6 | 115 hp (86 kW) @ 3400 RPM | 215 lb⋅ft (292 N⋅m) @ 1600 RPM | -
row 24 : 1985–1987 | 6.2 L Detroit Diesel V-8 | 148 hp (110 kW) @ 3600 RPM | 246 lb⋅ft (334 N⋅m) @ 2000 RPM | over 8500# GVWR
*/
Question: What is the horse power of the Chevrolet C/K engines through the years?
Please provide the answer in a fluent sentence: The Chevrolet C/K engines were a 160 hp (119 kW) 4.3 L V6, a 210 hp (157 kW) 5.7 L V8 and a 6.2 L diesel V8.

/*
Table page title: 1966 PGA Tour
Table section title: Money leaders
col : Rank | Player | Country | Earnings ($)
row 1 : 1 | Billy Casper | United States | 121,945
row 2 : 2 | Jack Nicklaus | United States | 111,419
row 3 : 3 | Arnold Palmer | United States | 110,468
row 4 : 4 | Doug Sanders | United States | 80,096
row 5 : 5 | Gay Brewer | United States | 75,688
row 6 : 6 | Phil Rodgers | United States | 68,360
row 7 : 7 | Gene Littler | United States | 68,345
row 8 : 8 | R. H. Sikes | United States | 67,349
row 9 : 9 | Frank Beard | United States | 66,041
row 10 : 10 | Al Geiberger | United States | 63,220
*/
Question: Who's the leading money winner and how much did he/she earn?
Please provide the answer in a fluent sentence: Casper was the leading money winner with earnings of $121,945.

/*
Table page title: McLeod Bethel-Thompson
Table section title: Statistics
col : Year | Team | Passing | Passing | Passing | Passing | Passing | Passing | Passing | Passing | Rushing | Rushing | Rushing | Rushing
row 1 : Year | Team | Cmp | Att | Pct | Yds | Y/A | TD | Int | Rtg | Att | Yds | Avg | TD
row 2 : 2007 | UCLA | 23 | 55 | 41.8 | 293 | 5.3 | 1 | 5 | 74.4 | 15 | 32 | 2.1 | 0
row 3 : 2008 | Sacramento | 28 | 49 | 57.1 | 415 | 8.5 | 2 | 1 | 137.7 | 16 | 33 | 2.1 | 1
row 4 : 2009 | Sacramento | 64 | 110 | 58.2 | 746 | 6.8 | 4 | 5 | 118.1 | 25 | 1 | 0.0 | 1
row 5 : 2010 | Sacramento | 21 | 38 | 55.3 | 161 | 4.2 | 1 | 2 | 89.0 | 6 | 5 | 0.8 | 0
row 6 : Career | Career | 136 | 252 | 54.0 | 1,615 | 6.4 | 8 | 13 | 108.0 | 62 | 3 | 0.0 | 2
*/
Question: How did Bethel do in 2007 for the team UCLA?
Please provide the answer in a fluent sentence: In 2007, Bethel completed 23 of 55 passes for 293 yards with one touchdown and five interceptions for UCLA.

/*
Table page title: List of Stanley Cup champions
Table section title: Active teams
col : Apps | Team | Wins | Losses | Win %
row 1 : 34 | Montreal Canadiens | 24 | 9 | .727
row 2 : 24 | Detroit Red Wings | 11 | 13 | .458
row 3 : 21 | Toronto Maple Leafs | 13 | 8 | .619
row 4 : 19 | Boston Bruins | 6 | 13 | .316
row 5 : 13 | Chicago Blackhawks | 6 | 7 | .462
*/
Question: How many Stanley Cup championships did Toronto Maple Leafs and Montreal Canadiens win?
Please provide the answer in a fluent sentence: The Toronto Maple Leafs has won thirteen Stanley Cup championships, and the Montreal Canadiens won 24 championships.

/*
Table page title: 2009 NHL Entry Draft
Table section title: Round one
col : Pick # | Player | Nationality | Position | NHL team | Team from | League from
row 1 : 1 | John Tavares | Canada | C | New York Islanders | London Knights | Ontario Hockey League
row 2 : 2 | Victor Hedman | Sweden | D | Tampa Bay Lightning | Modo Hockey | Elitserien (Sweden)
row 3 : 3 | Matt Duchene | Canada | C | Colorado Avalanche | Brampton Battalion | Ontario Hockey League
row 4 : 4 | Evander Kane | Canada | LW | Atlanta Thrashers | Vancouver Giants | Western Hockey League
*/
Question: Who were the second and third picks and what teams were they drafted by?
Please provide the answer in a fluent sentence: The Tampa Bay Lightning drafted the second pick defenceman Victor Hedman from Modo Hockey of the SEL, and the Colorado Avalanche drafted the third pick Matt Duchene of the Brampton Battalion.

/*
Table page title: EMD E-unit
Table section title: Models
col : Model designation | Build year | Total produced | AAR wheel arrangement | Prime mover | Power output | Image
row 1 : EA/EB | 1937-1938 | 6 A units, 6 B units | A1A-A1A | Winton 201-A | 1,800 hp (1,300 kW) | -
row 2 : E1 | 1937-1938 | 8 A units, 3 B units | A1A-A1A | Winton 201-A | 1,800 hp (1,300 kW) | Golden Gate Santa Fe train.JPG
row 3 : E2 | 1937 | 2 A units, 4 B units | A1A-A1A | Winton 201-A | 1,800 hp (1,300 kW) | City of Los Angeles 1944.JPG
row 4 : E3 | 1938-1940 | 17 A units, 2 B units | A1A-A1A | EMD 567 | 2,000 hp (1,490 kW) | -
row 5 : E4 | 1938-1939 | 14 A units, 5 B units | A1A-A1A | EMD 567 | 2,000 hp (1,490 kW) | -
row 6 : E5 | 1940-1941 | 11 A units, 5 B units | A1A-A1A | EMD 567 | 2,000 hp (1,490 kW) | -
row 7 : E6 | 1939-1942 | 91 A units, 26 B units | A1A-A1A | EMD 567 | 2,000 hp (1,490 kW) | -
row 8 : E7 | 1945-1949 | 428 A units, 82 B units | A1A-A1A | EMD 567A | 2,000 hp (1,490 kW) | Afternoon Hiawatha 1956.JPG
row 9 : E8 | 1949-1954 | 450 A units, 46 B units | A1A-A1A | EMD 567B | 2,250 hp (1,678 kW) | -
row 10 : E9 | 1954-1964 | 100 A units, 44 B units | A1A-A1A | EMD 567C | 2,400 hp (1,790 kW) | -
*/
Question: What were the prime movers and power outputs of the E8 and E9 models?
Please provide the answer in a fluent sentence: The E8 had V567B engines (2,250 hp (1.68 MW)), while the E9 had 567C engines (2,400 hp).

/*
Table page title: Williamsport Regional Airport
Table section title: Runways
col : Runway | Length | Notes
row 1 : 9/27 | 6,825 feet (2,080 m) | Longest runway used by nearly all commercial flights, equipped with ILS on 27 side.
row 2 : 12/30 | 4,273 feet (1,302 m) | -
row 3 : 33/15 | 2,300 feet (700 m) | Closed in 1979 and removed in 1981.
*/
Question: What runways of Williamsport Regional Airport are currently active?
Please provide the answer in a fluent sentence: Williamsport Regional Airport has two active runways, the longest (9/27) being 6,825 feet (2,080m), and 12/30 being 4,273 feet (1,302m)."""

query_tabfact = """/*
table caption : 2008 sidecarcross world championship.
col : position | driver / passenger | equipment | bike no | points
row 1 : 1 | daniël willemsen / reto grütter | ktm - ayr | 1 | 531
row 2 : 2 | kristers sergis / kaspars stupelis | ktm - ayr | 3 | 434
row 3 : 3 | jan hendrickx / tim smeuninx | zabel - vmc | 2 | 421
row 4 : 4 | joris hendrickx / kaspars liepins | zabel - vmc | 8 | 394
row 5 : 5 | marco happich / meinrad schelbert | zabel - mefo | 7 | 317
*/
Statement: bike number 3 is the only one to use equipment ktm - ayr.
The anwser is: NO

/*
table caption : 1957 vfl season.
col : home team | home team score | away team | away team score | venue | crowd | date
row 1 : footscray | 6.6 (42) | north melbourne | 8.13 (61) | western oval | 13325 | 10 august 1957
row 2 : essendon | 10.15 (75) | south melbourne | 7.13 (55) | windy hill | 16000 | 10 august 1957
row 3 : st kilda | 1.5 (11) | melbourne | 6.13 (49) | junction oval | 17100 | 10 august 1957
row 4 : hawthorn | 14.19 (103) | geelong | 8.7 (55) | brunswick street oval | 12000 | 10 august 1957
row 5 : fitzroy | 8.14 (62) | collingwood | 8.13 (61) | glenferrie oval | 22000 | 10 august 1957
*/
Statement: collingwood was the away team playing at the brunswick street oval venue.
The anwser is: NO

/*
table caption : co - operative commonwealth federation (ontario section).
col : year of election | candidates elected | of seats available | of votes | % of popular vote
row 1 : 1934 | 1 | 90 | na | 7.0%
row 2 : 1937 | 0 | 90 | na | 5.6%
row 3 : 1943 | 34 | 90 | na | 31.7%
row 4 : 1945 | 8 | 90 | na | 22.4%
row 5 : 1948 | 21 | 90 | na | 27.0%
*/
Statement: the 1937 election had a % of popular vote that was 1.4% lower than that of the 1959 election.
The anwser is: NO

/*
table caption : 2003 pga championship.
col : place | player | country | score | to par
row 1 : 1 | shaun micheel | united states | 69 + 68 = 137 | - 3
row 2 : t2 | billy andrade | united states | 67 + 72 = 139 | - 1
row 3 : t2 | mike weir | canada | 68 + 71 = 139 | - 1
row 4 : 4 | rod pampling | australia | 66 + 74 = 140 | e
row 5 : t5 | chad campbell | united states | 69 + 72 = 141 | + 1
*/
Statement: phil mickelson was one of five players with + 1 to par , all of which had placed t5.
The anwser is: YES"""


def simple_query(sample, table_info, llm, debug=False, use_demo=False, llm_options=None):
    table_text = table_info["table_text"]

    caption = sample["table_caption"]
    statement = sample["statement"]

    prompt = ""
    if DATASET == "tabfact":
        prompt += "Here are the statement about the table and the task is to tell whether the statement is True or False.\n"
        prompt += "If the statement is true, answer YES, and otherwise answer NO.\n"
    elif DATASET == "fetaqa":
        prompt += "Answer the following question based on the table. Please provide your answer in a fluent sentence.\n"
    elif DATASET == "wiki":
        prompt += "Here are the statements about the table. The task is to give a single answer based on the table. Just say the answer directly, without any explanation."
    if use_demo:
        prompt += "\n"
        prompt += eval(f"query_{DATASET}") + "\n\n"
        if DATASET == "tabfact":
            prompt += "Here are the statement about the table and the task is to tell whether the statement is True or False.\n"
            prompt += "If the statement is true, answer YES, and otherwise answer NO.\n"
        elif DATASET == "fetaqa":
            prompt += "Answer the question based on the table. Please provide your answer in a fluent sentence.\n"
        elif DATASET == "wiki":
            prompt += "Here are the statements about the table. The task is to give a single answer based on the table. Just say the answer directly, without any explanation."
        prompt += "\n"

    prompt += "/*\n"
    prompt += table2string(table_text, caption=caption) + "\n"
    prompt += "*/\n"

    if "group_sub_table" in table_info:
        group_column, group_info = table_info["group_sub_table"]
        prompt += "/*\n"
        prompt += "Group the rows according to column: {}.\n".format(group_column)
        group_headers = ["Group ID", group_column, "Count"]
        group_rows = []
        for i, (v, count) in enumerate(group_info):
            if v.strip() == "":
                v = "[Empty Cell]"
            group_rows.append([f"Group {i+1}", v, str(count)])
        prompt += " | ".join(group_headers) + "\n"
        for row in group_rows:
            prompt += " | ".join(row) + "\n"
        prompt += "*/\n"

    prompt += "Statement: " + statement + "\n"

    prompt += "The answer is:"
    responses = llm.generate_plus_with_score(prompt, options=llm_options)
    responses = [(res.strip(), np.exp(score)) for res, score in responses]

    if debug:
        print(prompt)
        print(responses)

    operation = {
        "operation_name": "simple_query",
        "parameter_and_conf": responses,
    }
    sample_copy = copy.deepcopy(sample)
    sample_copy["chain"].append(operation)

    return sample_copy

