#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  Push Video Updates to Pinecone and Algolia
"""

from typing import *

import os, json
from dotenv import load_dotenv


from json import JSONDecodeError

from pinecone import Pinecone, ServerlessSpec

from database import DbClient
from search import AlgoliaSearchClient
from utils import flatten_text

import torch
import random

from pymongo import MongoClient

import numpy as np

# db_client = DbClient()
search_client = AlgoliaSearchClient(index_name="discite-search")

# connect to MongoDB
load_dotenv()

# get PINECONE_API_KEY from .env file
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)

# use preTechnigalaClean_db
db = client.preTechnigalaClean_db

# count records in 'inference_summary' collection
total_records = db.inference_summary.count_documents({})

added_records = 0

topic_records = []
transcript_records = []

# load topicssemantic.json
# with open("topicssemantic.json", "r") as f:
#     topics = json.load(f)
    
# print(f"{topics = }")

# iterate over collections in 'inference_summary' collection
for inference_summary in db.inference_summary.find():
    # print(inference_summary)
    
    id = inference_summary["_id"]
    summary = inference_summary["inferenceSummary"]
    
    # print(f"{id}: {summary}")
    
    # get corresponding record from "video_metadata" collection
    video_metadata = db.video_metadata.find_one({ "_id": id })
    # print(f"\n\n{video_metadata = }\n\n")
    
    try:
        complexity = np.mean(video_metadata.get("inferenceComplexities", [ 0 ]))
    except Exception as e:
        complexity = 0
        print(f"WRONG: {video_metadata.get("inferenceComplexities", [ 0 ])}")
    
    search_data = {
        "objectID": f"{id}",                    # stringify
        "topics": [item["name"] for item in summary.get("generalTopics", []) ],
        "complexity": complexity,
        "uploader": video_metadata.get("uploader", ""),
    }
    
    topic_records.append(search_data)
    
    transcript_records.append({
        **search_data,
        "transcript": flatten_text(summary)
    })
    
try:
    print(f"ADDING: {len(topic_records)} records")
    
    print(f"TOPICS...")
    search_client.add_topic_records(topic_records)
    
    print(f"TRANSCRIPTS...")
    search_client.add_records(transcript_records)
    
except Exception as e:
    print(f"ID: {id}\nERROR: {e}")
    
print(f"ADDED: {len(topic_records)} records")
    

# PATH = "inference/data/outputs/101"
# SUMMARY_PATH = f"{PATH}/text"
# EMBEDDINGS_PATH = f"{PATH}/embeddings"

# topics = [
#     "Java", "JavaScript", "Python", "Backend", "FrontEnd",
#     "Machine Learning", "Deep Learning", "Data Science", "Data Engineering",
#     "SQL", "NoSQL", "Databases", "Cloud", "AWS", "GCP", "Azure",
# ]

# search_client.clear_index()
# invalid_count = 0
# for file in os.listdir(SUMMARY_PATH):
#     with open(f"{SUMMARY_PATH}/{file}", "r") as f:
#         try:
#             meta = json.load(f)
#             print(f"FILE:              {file} LOADED")
            
#             """
#                 Add search data to Algolia
#             """
#             search_data = {}
#             search_data["objectID"] = random.randint(0, 100000)
#             search_data["topics"] = [ random.choice(topics) for _ in range(3) ]
            
#             search_client.add_topic_record(search_data)
            
#             search_data["transcript"] = flatten_text(meta)
#             response = search_client.add_record(search_data)
            
#             print(f"RESPONSE:          {response}")
            
#             """
#                 Add video data to Pinecone
#             """
            
#             # load tensors

#             # get filename withouth extension
#             basename = os.path.splitext(file)[0]
#             with open(f"{EMBEDDINGS_PATH}/max_{basename}.pt", "rb") as f:
#                 tensor = torch.load(f, map_location="cpu", weights_only=True)
#                 tensor = tensor.numpy().tolist()
#                 print(f"TENSOR:            {len(tensor)}")
                
#                 # add to pinecone
#                 response = db_client.add_video(f"{search_data['objectID']}", embeddings=tensor, topics=search_data["topics"])
            
            
            
            
            
#             # print(f"FILE: {file}\nMETA: {meta}")
#         except JSONDecodeError as e:
#             print(f"FILE:              {file} ERROR: {e}")
#             invalid_count += 1
            
# print(f"INVALID COUNT: {invalid_count}")
