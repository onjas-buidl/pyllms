import pandas as pd

import llms
import datetime
import numpy as np
from pymongo import MongoClient
import dotenv, os
from bson.objectid import ObjectId
import tqdm

dotenv.load_dotenv()
OPENAI_API_KEY_LOCAL = os.getenv("OPENAI_API_KEY_LOCAL")

gpt4evaluator = llms.init('martian/openai/chat/gpt-4-turbo-128k', openai_api_key=OPENAI_API_KEY_LOCAL)


def parse_array_to_dict(arr):
    result = {}
    for item in arr:
        if item.startswith("MartianProvider"):
            # Extracting provider name
            provider_name = item.split("'")[1]
            result["Provider"] = provider_name
        elif item.startswith("Total Tokens"):
            # Extracting total tokens
            total_tokens = int(item.split(":")[1].strip())
            result["Total Tokens"] = total_tokens
        elif item.startswith("Total Cost"):
            # Extracting total cost
            total_cost = float(item.split(":")[1].strip())
            result["Total Cost"] = total_cost
        elif item.startswith("Median Latency"):
            # Extracting median latency
            median_latency = float(item.split(":")[1].strip())
            result["Median Latency"] = median_latency
        elif item.startswith("Aggregated speed"):
            # Extracting aggregated speed
            aggregated_speed = float(item.split(":")[1].strip())
            result["Aggregated Speed"] = aggregated_speed
        elif item.startswith("Accuracy"):
            # Extracting accuracy
            accuracy = float(item.split(":")[1].replace('%', '').strip())
            result["Accuracy"] = accuracy
    return result


def calculate_total_cost(start_datetime,
                         end_datetime,
                         other_query={},
                         api_key_id="64f782288d8f981dff8b5102",
                         test_db=False):
    dotenv.load_dotenv()
    CONNECTION_STRING = os.getenv("CONNECTION_STRING")
    CONNECTION_STRING_TEST = os.getenv("CONNECTION_STRING_TEST")
    if test_db:
        client = MongoClient(CONNECTION_STRING_TEST)
        api_key_id = ObjectId('6536e3ba2a6e555fdac842e4')
    else:
        client = MongoClient(CONNECTION_STRING)
        api_key_id = ObjectId(api_key_id)

    db = client["backend"]
    txs = db.transactions_v2

    query = {
        "api_key_id": api_key_id,
        "start_time": {
            "$gte": start_datetime,
            "$lte": end_datetime
        }
    }

    # Execute the query and calculate the total cost
    results = txs.find(query)
    results_list = [i for i in results]
    if len(other_query) > 0:
        for key, value in other_query.items():
            results_list = [i for i in results_list if i.get(key, '') == value]


    total_cost_sum = sum(result.get('total_cost', 0) for result in results_list)

    return total_cost_sum

"""
results = txs.find(query)
l = [i for i in results]
for item in l:
    print(item['total_cost'])
"""
total_result = []

# %% test all model performance

model_name_strings = [
    "anthropic/claude-instant-v1",
    # "anthropic/claude-v2", "meta/llama-2-70b-chat",
    # 'openai/chat/gpt-4-turbo-128k','openai/chat/gpt-4',
    'openai/chat/gpt-3.5-turbo',
    # 'router',
    'mistralai/mixtral-8x7b-chat'
]
model_name_strings = ['martian/' + model_name_string for model_name_string in model_name_strings]


for model_name_string in tqdm.tqdm(model_name_strings):
    print(model_name_string, end='\n\n')
    for _ in range(5):
        model_to_test = llms.init(model_name_string)
        start_datetime = datetime.datetime.utcnow()
        t = model_to_test.benchmark(evaluator=gpt4evaluator)
        end_datetime = datetime.datetime.utcnow()

        arr = t.get_csv_string().replace('\r','').split('\n')[-2].split(',')
        parsed_dict = parse_array_to_dict(arr)
        parsed_dict['model_name'] = model_name_string
        parsed_dict['start_datetime'] = start_datetime
        parsed_dict['end_datetime'] = end_datetime
        if 'router' in model_name_string:
            parsed_dict['calculated_cost'] = calculate_total_cost(start_datetime,
                                                                end_datetime,
                                                                other_query={'call_type': 'knn-classification'})
        else:
            parsed_dict['calculated_cost'] = calculate_total_cost(start_datetime,
                                                              end_datetime,
                                                              other_query={'chosen_model': model_name_string.replace('martian/', '')})
        # remote "Total Cost" from the dict
        parsed_dict.pop('Total Cost', None)
        total_result.append(parsed_dict)
        print(parsed_dict)

df = pd.DataFrame(total_result)
# get a datetime string for date and hour and minute of the current time to save as file name
datetime_string = datetime.datetime.now().strftime("%m-%d-%H-%M")

df.to_csv(f'results/{datetime_string}_eval.csv', index=False)


# %% visualize the result
import seaborn as sns
import pandas as pd

df = pd.read_csv('/Users/jasonmartian/Desktop/pyllms/results/12-15-23-14_eval.csv')
"""
>>> df.columns
Index(['Provider', 'Total Tokens', 'Median Latency', 'Aggregated Speed',
       'Accuracy', 'model_name', 'start_datetime', 'end_datetime',
       'calculated_cost'],
      dtype='object')
"""
df['Provider'] = df['model_name'].str.replace('martian/', '').replace('mistralai/mixtral-8x7b-chat', 'router')
# filter out gpt-3.5-turbo
df = df[df['model_name'] != 'martian/anthropic/claude-instant-v1']


df.Accuracy = df.Accuracy.astype(float) / 100
import matplotlib.pyplot as plt

scatter = sns.scatterplot(data=df, x='calculated_cost', y='Accuracy', hue='Provider', palette='rainbow', s=100)
plt.show()

#
# # Create a list of unique providers
# providers = df['Provider'].unique()
#
# # Create a colormap
# colors = plt.cm.rainbow(np.linspace(0, 1, len(providers)))
# fig, ax = plt.subplots(figsize=(10, 6))
# for i, provider in enumerate(providers):
#     # Filter data for the current provider
#     data = df[df['Provider'] == provider]
#
#     # Create a scatter plot for the current provider
#     ax.scatter(data['calculated_cost'], data['Accuracy'], color=colors[i], label=provider)
#
# plt.xlabel('Accuracy')
# plt.ylabel('Cost')
# plt.title('Average Accuracy vs. Average Cost by Provider')
#
# # Add a legend
# # plt.legend(bbox_to_anchor=(.2,.7), loc='upper left')
# plt.show()