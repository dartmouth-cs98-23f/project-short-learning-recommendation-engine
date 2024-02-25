#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from youtube_transcript_api import YouTubeTranscriptApi
import os
import csv
  
def download_transcript(id: str, filename: str) -> None:
  
  if os.path.exists(filename):
    print(f"Transcript {filename} already downloaded")
    return

  transcript = YouTubeTranscriptApi.get_transcript(id, languages=('en', 'en-US', 'en-GB'))
  print(f"Downloading transcript for {id} to {filename}")
  with open(filename, "w") as f:
    for line in transcript:
      stripped = line['text']
      f.write(f'{stripped}\n')
    
    print(f"Downloaded transcript for {id} to {filename}")
    return
