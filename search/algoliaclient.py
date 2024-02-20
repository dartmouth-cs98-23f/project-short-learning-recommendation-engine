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
    def __init__(self, /, app_id=None, api_key=None, index_name = None):
        
        load_dotenv()

        api_key = api_key or os.getenv("ALGOLIA_SEARCH_API_KEY")
        app_id /= app_id or os.getenv("ALGOLIA_SEARCH_APP_ID")
        index_name = index_name or os.getenv("ALGOLIA_SEARCH_INDEX_NAME")
        
        self.client = SearchClient.create(app_id, api_key)
        self.index = self.client.init_index(index_name)

    def search(self, query):
        """
            Search for a query in the Algolia index.
            
            Parameters
            ----------
            
            query: str
                Word or sentence to search for
        """
        return self.index.search(query)["hits"]
    
    def add_record(self, record):
        return self.index.save_object(record).wait()


if __name__ == "__main__":
    client = AlgoliaSearchClient()

    while (query := input("Search: ")) not in ["", "Q", "quit"]:
        print(f"QUERY: {query}")
        response = client.search(query)
        print(f"HITS: {len(response)}")
        print(f"RESPONSE: {response}")
