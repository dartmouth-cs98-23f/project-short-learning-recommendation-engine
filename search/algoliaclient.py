#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    search/algoliasearch.py
    ~~~~~~~~~~~~~~~~~~~~~~~
    
    A simple search client for Algolia.
"""

from algoliasearch.search_client import SearchClient
from dotenv import load_dotenv
import os


class AlgoliaSearchClient:
    """
        A wrapper around Algolia's search API.
        
        NOTE: Expects the following environment variables to be set:
            - ALGOLIA_SEARCH_API_KEY
            - ALGOLIA_SEARCH_APP_ID
            - ALGOLIA_SEARCH_INDEX_NAME
            
        Alternatively, you can pass these values to the constructor.
            
        Exposed Methods
        ---------------
        - search(query: str) -> dict
            Search for a query in the Algolia index.
        - add_record(record: dict) -> dict
            Add a record to the Algolia index.
            
        Example
        -------
        >>> # adding a record
        >>> record = {
        ...     "name": "Python",
        ...     "description": "A programming language."
        ... }
        >>> response = client.add_record(record)
        >>> print(response)
        >>>
        >>> # searching for a query
        >>> client = AlgoliaSearchClient()
        >>> response = client.search("Python")
        >>> print(response)
        
    """
    def __init__(self, /, app_id=None, api_key=None, index_name = None, topics_index_name = None):
        
        load_dotenv()

        api_key = api_key or os.getenv("ALGOLIA_SEARCH_API_KEY")
        app_id = app_id or os.getenv("ALGOLIA_SEARCH_APP_ID")
        index_name = index_name or os.getenv("ALGOLIA_SEARCH_INDEX_NAME")
        topics_index_name = os.getenv("ALGOLIA_SEARCH_TOPICS_INDEX_NAME")
        
        self.client = SearchClient.create(app_id, api_key)
        self.index = self.client.init_index(index_name)
        self.topics_index = self.client.init_index(topics_index_name)

    def search(self, query):
        """
            Search for a query in the Algolia index.
            
            Parameters
            ----------
            
            query: str
                Word or sentence to search for
                
            Returns
            -------
            
            dict
                Search results
        """
        return self.index.search(query)["hits"]
    
    def add_record(self, record):
        """
            Add new record in the Algolia index.
            
            Parameters
            ----------
            
            record: dict
                Collection of the record to be added.
                
            Returns
            -------
            
            Response status
        """
        return self.index.save_object(record).wait().raw_responses
    
    def add_records(self, records):
        """
            Add new records (bulk) to the Algolia index.
            
            Parameters
            ----------
            
            record: dict
                Collection of the record to be added.
                
            Returns
            -------
            
            Response status
        """
        return self.index.save_objects(records).wait().raw_responses
    
    def add_topic_record(self, record):
        """
            Add new record in the Algolia index.
            
            Parameters
            ----------
            
            record: dict
                Collection of the record to be added.
                
            Returns
            -------
            
            Response status
        """
        return self.topics_index.save_object(record).wait().raw_responses
    
    def add_topic_records(self, records):
        """
            Add new records (bulk) to the Algolia index.
            
            Parameters
            ----------
            
            record: dict
                Collection of the record to be added.
                
            Returns
            -------
            
            Response status
        """
        return self.topics_index.save_objects(records).wait().raw_responses
    
    def clear_transcripts(self):
        """
            Clear the Algolia index.
            
            Parameters
            ----------
            
            NONE
            
            Returns
            -------
            
            Response status
        """
        return self.index.clear_objects().wait().raw_responses
    
    def clear_topics(self):
        """
            Clear the Algolia index.
            
            Parameters
            ----------
            
            NONE
            
            Returns
            -------
            
            Response status
        """
        return self.topics_index.clear_objects().wait().raw_responses
    
    def clear(self):
        """
            Clear the Algolia index.
            
            Parameters
            ----------
            
            NONE
            
            Returns
            -------
            
            Response status
        """
        try:
            
            self.index.clear_objects().wait().raw_responses
            self.topics_index.clear_objects().wait().raw_responses
            
            print(f"CLEARED INDEXES: {self.index.index_name}, {self.topics_index.index_name}")
            
        except Exception as e:
            print(f"ERROR: {e}")


if __name__ == "__main__":
    client = AlgoliaSearchClient()

    while (query := input("Search: ")) not in ["", "Q", "quit"]:
        print(f"QUERY: {query}")
        response = client.search(query)
        print(f"HITS: {len(response)}")
        print(f"RESPONSE: {response}")
