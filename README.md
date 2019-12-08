# TemporalTracker
Kernel Correlation based tracker with Kalman smoother.

## Requirements:
 - Python 3+
 - (See requirements.txt file for python packages.)
 
## Installation
`pip install -r requirements.txt`

## Background

This is a correlation filter which tracks a target throughout a video sequence. Test video files were collected from:
 - Ball Test Media - https://www.youtube.com/watch?v=zsdPYFPTdw0
 - Others - http://www.visual-tracking.net
 
 ## Usage
 
 `python temporal_tracker.py --test_case redteam`
