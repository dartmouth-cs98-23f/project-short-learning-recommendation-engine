from pinecone import Pinecone, ServerlessSpec

from dotenv import load_dotenv
import os

class DbClient:
    """
        Database class to handle all database operations.
    """
    def __init__(self):
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

    def get_index(self, index_name):
        """
            Get an index from Pinecone.
        """
        # if index does not exist, create it
        print(f"INDICES: {self.client.list_indexes()}")
        if index_name not in self.client.list_indexes():
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
            
        return self.client.Index(index_name)
    
    def add_video(self, video_id: str, embedding):
        """
            Add a video to the Pinecone index.
        """
        index = self.get_index("recommendations")
        index.upsert(
            vectors = {
                "id": video_id,
                "values": embedding
            },
            namespace="video-transcripts"
        )
        
        print(f"INSERT: {video_id} - {embedding}")
        
    def get_video(self, video_id: str):
        """
            Get a video from the Pinecone index.
        """
        index = self.get_index("recommendations")
        return index.get(video_id)
