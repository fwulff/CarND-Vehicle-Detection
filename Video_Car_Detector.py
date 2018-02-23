# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:20:18 2017

@author: florian
"""

# Video Lane Detector
from moviepy.video.io.VideoFileClip import VideoFileClip
from pipeline import Detect_pipeline

# minimum number of detections
min_thresh = 3
max_thresh = 30

# reduction of heatmap per iteration
cooling_factor = 1

if __name__ == '__main__':
    
    #Initialise pipeline as object, including classifier, feature functions and window functions
    car_detector = Detect_pipeline(min_threshold = min_thresh , max_threshold = max_thresh, cooling_factor = cooling_factor)
    
    #read in video clip
    #video = VideoFileClip("videos/project_video.mp4")
    video = VideoFileClip("videos/test_video.mp4")

    #process a single frame, search sliding windows, classify, track with heatmap and visualize bounding boxes
    video_output = video.fl_image(car_detector.process_frame)
    
    # output name
    #video_result = "videos/project_video" + '_result.mp4'
    video_result = "videos/test_video" + '_result.mp4'
    
    #save result to file
    video_output.write_videofile(video_result, audio=False)