#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Video Ranking Utils
"""

class VideoRanker:
    """
        Video ranking class.
    """
    def __init__(self):
        """
            Constructor.
        """
        pass

    def get_user_affinities(self, user_id: str):
        """
            Get user affinities.
        """
        return {}

    
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
