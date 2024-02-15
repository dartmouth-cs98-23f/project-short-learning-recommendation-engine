#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Mainentrypoint into our recommentation engine.
"""

from fastapi import FastAPI
from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
import os

app = FastAPI()

# load .env file
load_dotenv()

print("PINECONE_API_KEY: ", os.getenv("PINECONE_API_KEY"))

# get PINECONE_API_KEY from .env file
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Create a Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

app.get("/")
def read_root():
    """
        Root endpoint.
    """
    return {"Hello": "World"}
