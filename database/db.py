from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
import os

import random

class DbClient:
    """
        Database class to handle all database operations.
    """
    def __init__(self, debug=False):
        """
            Constructor for the Database class.
        """
        # load .env file
        load_dotenv()

        print("PINECONE_API_KEY: ", os.getenv("PINECONE_API_KEY"))

        # get PINECONE_API_KEY from .env file
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

        # Create a Pinecone client
        self.client = Pinecone(api_key=PINECONE_API_KEY)
        
        self.debug = debug

    def get_index(self, index_name):
        """
            Get an index from Pinecone.
        """
        # if index does not exist, create it
        if self.debug:
            print(f"INDICES: {self.client.list_indexes()}")
        
        try:
            index = self.client.Index(index_name)
            
        except:
            if self.debug:
                print(f"Index {index_name} does not exist. Creating it now.")
            self.client.create_index(
                name = index_name,
                dimension = 8,
                metric = "cosine",
                spec = ServerlessSpec(
                    cloud = "aws",
                    region = "us-west-2"
                )
            )
            
            index = self.client.Index(index_name)
        
        return index
    
    def add_video(self, video_id: str, embeddings=None, topics=None):
        """
            Add a video to the Pinecone index.
        """
        index = self.get_index("recommendations")
        embeddings = embeddings or self.get_embeddings(video_id)
        topics = topics or []
        response = index.upsert(
            vectors = [{
                "id": video_id,
                "values": embeddings,
                "metadata": {
                    "topics": topics
                }
            }],
            namespace="video-transcripts",
        )
        
        if self.debug:
            print(f"{response}: INSERT {video_id} - {embeddings}")
            
        return response.to_dict()
        
    def get_video(self, video_id: str):
        """
            Get a video from the Pinecone index.
        """
        index = self.get_index("recommendations")
        result = index.query( id=video_id, top_k=1, namespace="video-transcripts")
        if self.debug:
            print(f"RESULT: {result.to_dict()}")
        return result.to_dict()
    
    def get_video_values(self, video_id: str):
        """
            Get the values for a video.
        """
        index = self.get_index("recommendations")
        result = index.query(id=video_id, top_k=1, namespace="video-transcripts", include_values=True)
        result = result.to_dict()
        if self.debug:
            print(f"RESULT: {result}")
        matches = result.get("matches", [])
        
        if not matches:
            return []
        
        return matches[0].get("values", [])
        
    
    
    def get_embeddings(self, video_id: str):
        """
            Get the embeddings for a video.
            
            Random for now. Replace with actual embeddings
            from Mixtral.
        """
        return [ random.random() for _ in range(8)]

    def get_recommendations(self, video_id: str, count=10):
        """
            Get video recommendations.
        """
        
        # get vectors for video_id
        values = self.get_video_values(video_id)
        if self.debug:
            print(f"VALUES: {values}")
        
        if not values:
            return {  }
        index = self.get_index("recommendations")
        result = index.query(
            namespace="video-transcripts",
            vector=values,
            top_k=count,
            include_values=False
        )
        
        return result.to_dict()
