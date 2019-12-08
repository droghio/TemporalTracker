# TemporalTracker
Correlation based tracker with a Kalman smoother.

## Requirements:
 - Python 3+
 - (See the [requirements.txt](requirements.txt) file for python packages.)
 
## Installation
`pip install -r requirements.txt`

## Background

The tracker uses a correlation filter to detect features of interest in a video frame. These are weighted and sent into a smoother to determine the tracked object's position. Test video files were collected from:
 - Ball Test Case - https://www.youtube.com/watch?v=zsdPYFPTdw0
 - Others - http://www.visual-tracking.net
 
Tracking is accomplished by maintaining a set of positive and negative correlation kernels. These are used to determine which regions of a frame most closely matches the object under track. These scores are fed into a weighting scheme which uses the predicted state of the Kalman to filter out background noise. The highest resulting score is classified as the target and used to update the state of the tracker.
 
The kernels are refined as subsequent frames are collected. The rate at which this occurs is defined by the tracker's learning rate. This and the Kalman's process noise dictate how flexible the tracker is to changing conditions verses susceptablility to track steals from clutter or other objects in the frame.

The tracker is implemented in Python and leverages the OpenCV library for image processing calls. The tracker runs in real time and comes with a series of test cases.

More information on the algorithm design can be found in the included [design presentation](doc/TemporalTracker.pdf) under the [doc](doc) directory. The PowerPoint version includes video generated from the tracker.
 
## Usage
 
 `python temporal_tracker.py --test_case redteam`
