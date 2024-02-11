#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Main entrypoint into our recommentation engine.
"""

from fastapi import FastAPI
from pinecone import Pinecone, ServerlessSpec

from database import DbClient

from dotenv import load_dotenv
import os

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
    
    db_client.add_video(video_id)
    
    # return inserted video data
    return db_client.get_video(video_id)

@app.get("/get-video/{video_id}")
def get_video_recommendatiosn(video_id: str):
    """
        Get video recommendations.
    """
    
    return db_client.get_recommendations(video_id, count=10)
