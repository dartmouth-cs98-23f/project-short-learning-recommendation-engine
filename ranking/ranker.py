#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Video Ranking Utils
"""

from dotenv import load_dotenv
import os
import requests

# import requests


class VideoRanker:
    """
        Video ranking class.
    """
    def __init__(self, /, backend_url = None, debug = False):
        
        load_dotenv()

        self.backend_url = backend_url or os.getenv("BACKEND_URL")
        self.debug = debug

    def get_user_affinities(self, user_id: str):
        """
            Get user affinities.
        """
        endpoint = f"{self.backend_url}/api/user/affinities/admin"
        payload = { "user_id": user_id }
        res = requests.get(endpoint, json=payload)
        
        if self.debug:
            print(f"GET USER AFFINITIES: {user_id} -> {res.json() = }")
        return res.json()

    
    def rank_videos(self, videos, user_id: str):
        """
            Rank videos.
        """
        #? get user affinities
        affinities = self.get_user_affinities(user_id)
        
        #? adjust weights based on user affinities
        for video in videos:
            """
                set new weight as distance (positive is better)
            """
            video.weight = video.score + affinities.get(video.topic, 0)
            
        #? rearrange based on weights
        videos.sort(key=lambda x: x.weight)
        
        return videos

if __name__ == "__main__":
    ranker = VideoRanker()
    print(ranker.get_user_affinities("user_1"))
    print(ranker.rank_videos([{"topic": "Python", "score": 0.5}, {"topic": "Python", "score": 0.7}], "user_1"))
