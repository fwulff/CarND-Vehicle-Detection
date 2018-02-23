# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:23:24 2017

@author: florian
"""

# Pipeline for car detection

from scipy.ndimage.measurements import label
import numpy as np
import cv2

from window_helper import draw_boxes
from classifier import car_classifier

# Class to include classifier and process single frame, bounding boxes and heatmap
class Detect_pipeline:
    
    def __init__(self, min_threshold = 1, max_threshold = 25, cooling_factor = 1):
        
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.cooling_factor = cooling_factor
        
        self.current_frame = 0
        self.hot_windows = None
        self.initialized = False
        self.window_img = None
        self.heatmap = None
        self.labels = None
        self.bb_image = None
        self.thresh_heatmap = None
        
        # classifier object including trained svm and single image classifier
        self.car_finder = car_classifier()
    
    # Function to initialise a blanc heatmap    
    def create_heatmap(self, image):
        self.initialized = True
        self.heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        self.thresh_heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Function to initialise Detect_pipeline class
    def init_pipeline(self, image):
        
        # train the classifier
        self.car_finder.train_classifier()
        # create the sliding windows
        self.car_finder.create_search_windows(image)
        # create the heatmap
        self.create_heatmap(image)
    
    # Function to increment heatmap with recent detections
    def add_heat(self, heatmap, boxlist):
        # Iterate through list of bboxes
        for box in boxlist:
            # Add += 1 for all pixels inside each bbox
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        self.heatmap = heatmap
    
    # Function to slowly reduce heatmap when no detections
    def reduce_heat(self, heatmap):
        #Do not set below zero
        heatmap = heatmap - self.cooling_factor
        print("Heatmap min ", np.min(self.heatmap))
        print("Heatmap max ", np.max(self.heatmap))
        self.heatmap = heatmap
        
    # Function to threshold the heatmap to suppress false detections
    def apply_threshold(self, heatmap, min_threshold = 1, max_threshold = 25):
        # Zero out pixels below the threshold
        heatmap[heatmap <= min_threshold] = 0
        heatmap[heatmap > max_threshold] = max_threshold
        # Return thresholded map
        self.thresh_heatmap = heatmap

    # Function to create the final result image by drawing bounding boxes according to labels of cars
    def draw_labeled_bboxes(self, image):
    # Iterate through all detected car
        self.bb_image = image
        
        for car_number in range(1, self.labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (self.labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(image, bbox[0], bbox[1], (0,0,255), 6)
            self.bb_image = image

    # function to classify and draw bounding boxes on a single image
    def find_windows(self, image):
        
        draw_image = np.copy(image)
        
        #create empty list of detected car windows
        self.hot_windows = []
        
        # classify all windows using the classifier and return detected cars
        self.hot_windows = self.car_finder.search_windows(draw_image)
        
        # draw detected bounding boxes to image
        self.window_img = draw_boxes(draw_image, self.hot_windows, color=(0, 0, 255), thick=6)   
        
    # Function to process one single image using the whole pipeline
    def process_frame(self, image):
        
        # save every single step to an image for debugging
        # count number of process frames for file names
        self.current_frame += 1
        #cv2.imwrite('test_single_images/test_%s.jpg' % self.current_frame, image)
        
        if self.initialized == False:
            # Function to initialise Detect_pipeline class
            self.init_pipeline(image)
            print("pipeline initialized = ", self.initialized)
        
        # Function to classify and draw bounding boxes on a single image
        self.find_windows(image)
        #cv2.imwrite('test_window_images/window_img_%s.jpg' % self.current_frame, self.window_img)
        
        # Function to increment heatmap with recent detections
        self.add_heat(self.heatmap, self.hot_windows)
        #final_map = np.clip((self.heatmap - 2)*10, 0, 255)
        #cv2.imwrite('test_heat_images/heat_img_%s.jpg' % self.current_frame, final_map)
            
        # Function to threshold the heatmap to suppress false detections
        self.apply_threshold(self.heatmap, self.min_threshold, self.max_threshold)
        #clipped_thresh_heatmap = np.clip((self.thresh_heatmap - 2)*10, 0, 255)
        #cv2.imwrite('test_thresh_heat_images/thresh_heat_img_%s.jpg' % self.current_frame, clipped_thresh_heatmap)
        
        # Use heatmap to create label / segmentation and identification of peaks
        self.labels = label(self.heatmap) # labels is [array size, Number of labels]
        print("Number of cars ", self.labels[1])
        
        # Function to slowly reduce heatmap when no detections
        self.reduce_heat(self.heatmap)
        
        # Function to create the final result image by drawing bounding boxes according to labels of cars
        self.draw_labeled_bboxes(image)
        #cv2.imwrite('test_bb_images/bb_img_%s.jpg' % self.current_frame, self.bb_image)
        
        # return final image with thresholded detections
        return self.bb_image
        
