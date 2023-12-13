import llms
import datetime
gpt4evaluator = llms.init('gpt-4')

# %% run gpt-3.5-turbo benchmarking
# model = llms.init('martian/gpt-3.5-turbo')
#
# start_datetime = datetime.datetime.utcnow()
# t = model.benchmark(evaluator=gpt4evaluator)
# end_datetime = datetime.datetime.utcnow()
#
# print(t)

# %% run router benchmarking
model = llms.init('martian/router')

start_datetime = datetime.datetime.utcnow()
t = model.benchmark(evaluator=gpt4evaluator)
end_datetime = datetime.datetime.utcnow()

print(t)

# %% calculate total cost by querying mongo db
from pymongo import MongoClient
import dotenv, os
dotenv.load_dotenv()
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
client = MongoClient(CONNECTION_STRING)
db = client["backend"]
txs = db.transactions_v2

# query by selecting the entry that have api_key_id = 656e4ca4003279bef6fe0acf and start_time between start_datetime and end_datetime
api_key_id = "656e4ca4003279bef6fe0acf"

# Assuming start_datetime and end_datetime are defined
start_datetime = datetime.datetime(2022, 1, 1)
end_datetime = datetime.datetime(2022, 12, 31)

from bson.objectid import ObjectId
query = {
    "api_key_id": ObjectId(api_key_id),
    "start_time": {
        "$gte": start_datetime,
        "$lte": end_datetime
    }
}

results = txs.find(query)

# Iterate over the results
for result in results:
    print(result)