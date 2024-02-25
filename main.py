#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Main entrypoint into our recommentation engine.
"""

from typing import *

from fastapi import FastAPI
from pinecone import Pinecone, ServerlessSpec

from database import DbClient
from search import AlgoliaSearchClient
<<<<<<< HEAD
from ranking import VideoRanker
=======
>>>>>>> main

app = FastAPI()
db_client = DbClient()
search_client = AlgoliaSearchClient(index_name="discite-search")
<<<<<<< HEAD
ranker = VideoRanker()
=======
>>>>>>> main

app.get("/")
def read_root():
    """
        Root endpoint.
    """
    return {"Hello": "World"}

@app.get("/add-video/{video_id}")
def add_video(video_id: str):
    """
        Add a video to the Pinecone index.
    """
    
    return db_client.add_video(video_id)

@app.get("/get-recommendations/{video_id}")
def get_recommendations(video_id: str):
    """
        Get video recommendations.
    """
    
    return db_client.get_recommendations(video_id, count=10)


@app.get("/search")
def search(query: str, user_id: Optional[str] = None):
    """
        Search for a query in the Algolia index.
    """
    
    results = search_client.search(query)
    
    #? if user provided, rerank based on user affinites
    if user_id:
        results = ranker.rank_videos(results, user_id)
    
    return results
