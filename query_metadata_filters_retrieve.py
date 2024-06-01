import ast
from pathlib import Path
from datetime import datetime
import json
from openai import OpenAI

QUERY_FILTERS_FILE = "query_metadata_filters.json"

EXTRACT_FILTER_TEMPLATE = """Some questions will be provided below. 
Given the question, extract the metadata to filter the database about article sources. Avoid stopwords.
-----------------------------------------------------------------------------
The sources can only be from the list: 
['Yardbarker', 'The Guardian', 'Revyuh Media', 'The Independent - Sports', 'Wired', 'Sport Grill', 
'Hacker News', 'Iot Business News', 'Insidesport', 'Sporting News', 'Seeking Alpha', 
'The Age', 'CBSSports.com', 'The Sydney Morning Herald', 'FOX News - Health', 
'Science News For Students', 'Polygon', 'The Independent - Life and Style', 'FOX News - Entertainment', 
'The Verge', 'Business Line', 'The New York Times', 'The Roar | Sports Writers Blog', 'Sportskeeda', 
'BBC News - Entertainment & Arts', 'Business World', 'BBC News - Technology', 'Essentially Sports', 
'Mashable', 'Advanced Science News', 'TechCrunch', 'Financial Times', 'Music Business Worldwide', 
'The Independent - Travel', 'FOX News - Lifestyle', 'TalkSport', 'Yahoo News', 
'Scitechdaily | Science Space And Technology News 2017', 'Globes English | Israel Business Arena', 
'Wide World Of Sports', 'Rivals', 'Fortune', 'Zee Business', 'Business Today | Latest Stock Market And Economy News India', 
'Sky Sports', 'Cnbc | World Business News Leader', 'Eos: Earth And Space Science News', 
'Live Science: The Most Interesting Articles', 'Engadget']
-----------------------------------------------------------------------------
Examples to follow:

Question: Who is the individual associated with the cryptocurrency industry facing a criminal trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, and is accused by prosecutors of committing fraud for personal gain?
Answer: {{'source': {{'$in': ['The Verge', 'TechCrunch']}}}}

Question: After the TechCrunch report on October 7, 2023, concerning Dave Clark's comments on Flexport, and the subsequent TechCrunch article on October 30, 2023, regarding Ryan Petersen's actions at Flexport, was there a change in the nature of the events reported?
Answer: {{'source': {{'$in': ['TechCrunch']}}, 'published_at': '$in': {{['October 7, 2023', 'October 30, 2023']}}}}

Question: Which company, known for its dominance in the e-reader space and for offering exclusive invite-only deals during sales events, faced a stock decline due to an antitrust lawsuit reported by 'The Sydney Morning Herald' and discussed by sellers in a 'Cnbc | World Business News Leader' article?
Answer: {{'source': {{'$in': ['The Sydney Morning Herald', 'Cnbc | World Business News Leader']}}}}
-----------------------------------------------------------------------------
If you detect multiple queries, return the answer for the first. Now it is your turn:

Question: {query}
Answer:
"""


def clean_filter(filter_dict: dict) -> dict:
    for filter_key in filter_dict.copy().keys():
        if filter_key not in ["source", "published_at"]:
            del filter_dict[filter_key]
    if "published_at" in filter_dict:
        if isinstance(filter_dict["published_at"], dict):
            for published_at in filter_dict["published_at"].copy().keys():
                if isinstance(filter_dict["published_at"][published_at], list):
                    for date in filter_dict["published_at"][published_at]:
                        try:
                            datetime.strptime(date, "%B %d, %Y")
                        except (ValueError, TypeError):
                            del filter_dict["published_at"]
                            break
                else:
                    del filter_dict["published_at"]
                    break
        else:
            del filter_dict["published_at"]
    return filter_dict


with open("MultiHop-RAG/dataset/MultiHopRAG.json") as query_data_f:
    query_data_list = json.load(query_data_f)

filename = Path(QUERY_FILTERS_FILE)

try:
    with open(filename) as query_filters_f:
        query_filters = json.load(query_filters_f)
    present_queries = [query_filter["query"] for query_filter in query_filters]
except FileNotFoundError:
    filename.touch(exist_ok=True)
    present_queries = []
    query_filters = []

client = OpenAI()

for index, query in enumerate(query_data_list):
    if query["query"] not in present_queries:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": EXTRACT_FILTER_TEMPLATE.format(query=query["query"]),
                }
            ],
            temperature=0.1,
        )
        filter_str = completion.choices[0].message.content

        # Sometimes LLM returns not a valid dict. Rerun stript, and it will pick up on latest query.
        # LLM will return a valid dict on next try.
        filter_dict = ast.literal_eval(filter_str)
        filter_dict = clean_filter(filter_dict)
        print(filter_dict)

        query_filters.append(
            {"query": query["query"], "filter": clean_filter(filter_dict)}
        )
        with open(filename, "w") as query_filters_f:
            json.dump(query_filters, query_filters_f, indent=4, sort_keys=True)
