#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Main entrypoint into our recommentation engine.
"""

from typing import *

def flatten_text(video_meta: Dict) -> str:
    """
        Flatten video metadata into a single string.
    """

    s = ""
    for section in video_meta.get("sections", []):
        s += "\n".join(section.get("content", []))
            
    return s
