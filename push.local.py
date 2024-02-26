#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  Push Video Updates to Pinecone and Algolia
"""

from typing import *
import os, json

from json import JSONDecodeError

from pinecone import Pinecone, ServerlessSpec

from database import DbClient
from search import AlgoliaSearchClient
from utils import flatten_text

import torch

import random

db_client = DbClient()
search_client = AlgoliaSearchClient(index_name="discite-search")

PATH = "inference/data/outputs/101"
SUMMARY_PATH = f"{PATH}/text"
EMBEDDINGS_PATH = f"{PATH}/embeddings"

topics = [
    "Java", "JavaScript", "Python", "Backend", "FrontEnd",
    "Machine Learning", "Deep Learning", "Data Science", "Data Engineering",
    "SQL", "NoSQL", "Databases", "Cloud", "AWS", "GCP", "Azure",
]

search_client.clear_index()
invalid_count = 0
for file in os.listdir(SUMMARY_PATH):
    with open(f"{SUMMARY_PATH}/{file}", "r") as f:
        try:
            meta = json.load(f)
            print(f"FILE:              {file} LOADED")
            
            """
                Add search data to Algolia
            """
            search_data = {}
            search_data["objectID"] = random.randint(0, 100000)
            search_data["topics"] = [ random.choice(topics) for _ in range(3) ]
            
            search_client.add_topic_record(search_data)
            
            search_data["transcript"] = flatten_text(meta)
            response = search_client.add_record(search_data)
            
            print(f"RESPONSE:          {response}")
            
            """
                Add video data to Pinecone
            """
            
            # load tensors

            # get filename withouth extension
            basename = os.path.splitext(file)[0]
            with open(f"{EMBEDDINGS_PATH}/max_{basename}.pt", "rb") as f:
                tensor = torch.load(f, map_location="cpu", weights_only=True)
                tensor = tensor.numpy().tolist()
                print(f"TENSOR:            {len(tensor)}")
                
                # add to pinecone
                response = db_client.add_video(f"{search_data['objectID']}", embeddings=tensor, topics=search_data["topics"])
            
            
            
            
            
            # print(f"FILE: {file}\nMETA: {meta}")
        except JSONDecodeError as e:
            print(f"FILE:              {file} ERROR: {e}")
            invalid_count += 1
            
print(f"INVALID COUNT: {invalid_count}")
