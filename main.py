#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Main entrypoint into our recommentation engine.
"""

from fastapi import FastAPI
from pinecone import Pinecone, ServerlessSpec

from database import DbClient

app = FastAPI()
db_client = DbClient()

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
