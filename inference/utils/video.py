#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  Video downloader
"""

import os
from pytube import YouTube
from yt_dlp import YoutubeDL

def download_video(url: str, filename: str) -> None:
  """
    Download a video from url to outfile.

    Args:
      - `url (str)` : URL of video (not ID!!)
      - `outfile (str)`: filename to save to

    Returns:
      - `None`

    Raises an exception: if error connecting or downloading
  """

  try:
    ydl_opts = {
        'format': 'best[ext=mp4]/best',  # Explicitly specify the best format (single file)
        'outtmpl': filename,
    }

    with YoutubeDL(ydl_opts) as ydl:
          ydl.download([url])
          print(f"Downloaded {filename} from {url}")
  except Exception as e:
    print(f"Error Connecting: {e}")
    raise e

def download_videos_from_file(file_path: str) -> None:
    """
    Download videos from URLs and filenames listed in a text file.

    Args:
      - `file_path (str)`: Path to the text file containing [url, file name] pairs

    Returns:
      - `None`
    """
    with open(file_path, 'r') as file:
        for line in file:
            url, topic, subtopic, title = line.strip().split(', ')
            topic = topic.replace(' ', '_')
            subtopic = subtopic.replace(' ', '_')
            title = title.replace(' ', '_')
            download_video(url, f"{topic}_{subtopic}_{title}")

if __name__ == "__main__":
    file_path = "videos_23f"  # Change this to your actual text file path
    download_videos_from_file(file_path)