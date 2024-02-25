from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from inference import vectorize_pipeline

import dotenv
import os

dotenv.load_dotenv(dotenv_path="../.env")
uri = os.getenv("MONGODB_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
                          
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your dceployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

database_name = "kevininfpipe"
collection_name = "video_metadata"

db = client[database_name]
collection = db[collection_name]

# if doc does not have a field called "isVectorized", add it and set it to False
# if doc does not have a field called "isClipped", add it and set it to False
# for doc in collection.find():
#     if "isVectorized" not in doc:
#         collection.update_one({"_id": doc["_id"]}, {"$set": {"isVectorized": False}})
#     if "isClipped" not in doc:
#         collection.update_one({"_id": doc["_id"]}, {"$set": {"isClipped": False}})
    
    
    
doc = collection.find_one({"isVectorized": False})

print(doc)

vectorize_pipeline(doc)