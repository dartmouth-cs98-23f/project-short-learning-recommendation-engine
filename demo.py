#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    A demo entrypoint into our recommentation engine.
    
    `main.py` should mirror this file.
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

@app.get("/test-add")
def test_add():
    """
        Test the Pinecone client.
    """
    
    try:
        index = pc.Index("quickstart")
    except:
        pc.create_index(
            
            name="quickstart",
            dimension=8,
            metric="euclidean",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-west-2'
            ) 
        )
        
        index = pc.Index("quickstart")


    index.upsert(
        vectors=[
            {"id": "vec1", "values": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]},
            {"id": "vec2", "values": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]},
            {"id": "vec3", "values": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]},
            {"id": "vec4", "values": [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]}
        ],
        namespace="ns1"
    )

    index.upsert(
        vectors=[
            {"id": "vec5", "values": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]},
            {"id": "vec6", "values": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]},
            {"id": "vec7", "values": [0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]},
            {"id": "vec8", "values": [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]}
        ],
        namespace="ns2"
    )
    
    index_stats = index.describe_index_stats()
    
    return {"index-stats": index_stats.to_dict()}

@app.get("/test-query")
def test_query():
    """
        Test the Pinecone client.
    """

    index = pc.Index("quickstart")

    v1 = index.query(
        namespace="ns1",
        vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        top_k=3,
        include_values=True
    )

    v2 = index.query(
        namespace="ns2",
        vector=[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
        top_k=3,
        include_values=True
    )
    
    return {"v1": v1.to_dict(), "v2": v2.to_dict()}

@app.get("/test-delete")
def test_delete():
    """
        Test the Pinecone client.
    """

    index = pc.Index("quickstart")

    index.delete(
        ids=["vec1", "vec2", "vec3", "vec4"],
        namespace="ns1"
    ) 

    index.delete(
        ids=["vec5", "vec6", "vec7", "vec8"],
        namespace="ns2"
    )
    
    return {"status": "success"}
