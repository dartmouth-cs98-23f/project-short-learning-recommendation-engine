#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  A few utilities for CLIP and BART
"""

from youtube_transcript_api import YouTubeTranscriptApi
import os
import csv
  
def download_transcript(id: str, filename: str) -> None:
  
  if os.path.exists(filename):
    print(f"Transcript {filename} already downloaded")
    return

  transcript = YouTubeTranscriptApi.get_transcript(id, languages=('en', 'en-US'))
  print(f"Downloading transcript for {id} to {filename}")
  with open(filename, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["second", "duration", "transcript"])
    for line in transcript:
      stripped = line['text'].strip(" \n'\"")
      writer.writerow([line['start'], line['duration'], stripped])
    
    print(f"Downloaded transcript for {id} to {filename}")
    return
