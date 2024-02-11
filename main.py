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
    
    embedding = [1, 2, 3, 4, 5, 6, 7, 8]
    db_client.add_video(video_id, embedding)
    
    # return inserted video data
    return db_client.get_video(video_id)
