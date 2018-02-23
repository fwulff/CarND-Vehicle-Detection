# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:25:53 2017

@author: florian
"""

# classifier

import gc
import numpy as np
import glob
import time
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

from feature_helper import extract_features, single_img_features
from window_helper import slide_window


class car_classifier:
    
    def __init__(self):
        #self.cars = []
        #self.notcars = []
        #self.X_train = []
        #self.X_test = []
        #self.y_train = []
        #self.y_test = []
    
        self.classifier = None
        self.X_scaler = None
        self.windows = []
                
        self.test_size=0.2
        
        ### TODO: Tweak these parameters and see how the results change.
        self.color_space='YUV'
        self.spatial_size=(32, 32) 
        self.hist_bins=32
        self.hist_range=(0, 256) 
        self.orient=9
        self.pix_per_cell=8
        self.cell_per_block=2
        self.hog_channel=0
        self.spatial_feat=True
        self.hist_feat=True
        self.hog_feat=True
        
        self.huge_x_start_stop=[None, None]
        self.huge_y_start_stop=[400, 720]
        self.huge_xy_window=(96, 96)
        self.huge_xy_overlap=[0.75, 0.75]
        
        self.large_x_start_stop=[None, None]
        self.large_y_start_stop=[400, 528]
        self.large_xy_window=(64, 64)
        self.large_xy_overlap=[0.6, 0.6]
        
        self.small_x_start_stop=[None, None]
        self.small_y_start_stop=[400, 528]
        self.small_xy_window=(48,48)
        self.small_xy_overlap=[0.5, 0.5]

    # train the classifier by reading the training data, create the feature vectors for both cars and notcars, create the scaler, split training and test data sets and train and test the classifier
    # only the final classifier and scaler are stored, not the data, to reduce memory allocation of the class    
    def train_classifier(self):
        
        # Read in cars and notcars from small dataset (64x64 jpeg format)
        # data from .jpg (scaled 0 to 255)
        
        images_small = glob.glob('datasets/smallsets/*/*/*.jpeg')
        
        #images = glob.glob('datasets/bigsets/*/*/*.png')
        cars = []
        notcars = []
        
        for image in images_small:
            if 'image' in image or 'extra' in image:
                notcars.append(image)
            else:
                cars.append(image)
        
        # Read in cars and notcars from big datasets (64x64 png format)
        # data from .png images (scaled 0 to 1 by mpimg)
        #image = image.astype(np.float32)/255

        """
        images_big_non_vehicles = glob.glob('datasets/bigsets/non-vehicles/*/*.png')
        for image in images_big_non_vehicles:
            self.notcars.append(image)
            
        images_big_vehicles = glob.glob('datasets/bigsets/vehicles/*/*.png')
        for image in images_big_vehicles:
            self.cars.append(image)
        """
        #check number of used images
        print("read in nb cars", len(cars))
        print("read in nb not_cars", len(notcars))
        print("read in total", len(notcars)+len(cars))
        
        #extract car feature l from all car images
        car_features = extract_features(cars, 
                                        color_space=self.color_space, 
                                        spatial_size=self.spatial_size, 
                                        hist_bins=self.hist_bins, 
                                        orient=self.orient, 
                                        pix_per_cell=self.pix_per_cell, 
                                        cell_per_block=self.cell_per_block, 
                                        hog_channel=self.hog_channel, 
                                        spatial_feat=self.spatial_feat, 
                                        hist_feat=self.hist_feat, 
                                        hog_feat=self.hog_feat)
                                        
        #extract car feature vector from all car images                                   
        notcar_features = extract_features(notcars, 
                                        color_space=self.color_space, 
                                        spatial_size=self.spatial_size, 
                                        hist_bins=self.hist_bins, 
                                        orient=self.orient, 
                                        pix_per_cell=self.pix_per_cell, 
                                        cell_per_block=self.cell_per_block, 
                                        hog_channel=self.hog_channel, 
                                        spatial_feat=self.spatial_feat, 
                                        hist_feat=self.hist_feat, 
                                        hog_feat=self.hog_feat)
         
        # Stack feature vector                 
        X = np.vstack((car_features, notcar_features)).astype(np.float64)     
                   
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)
        
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)
        
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, 
                                                            y, 
                                                            test_size=self.test_size, 
                                                            random_state=rand_state)
        
        # Print statistics about data set
        print('Using:', 
              self.orient,'orientations', 
              self.pix_per_cell, 'pixels per cell and', 
              self.cell_per_block,'cells per block')
            
        print('Feature vector length:', len(X_train[0]))
        
        # Use a linear SVC as classifier
        self.classifier = LinearSVC()
        
        # Check the training time for the SVC
        t=time.time()
        
        # train classifier
        self.classifier.fit(X_train, y_train)
        t2 = time.time()
        
        print(round(t2-t, 2), 'Seconds to train SVC...')
        
        # Check the score of the SVC using the test set
        print('Test Accuracy of SVC = ', round(self.classifier.score(X_test, y_test), 4))

        #Free Memory
        del t, t2, images_small, cars, notcars, image, car_features, notcar_features, X, scaled_X, y, rand_state, X_train, X_test, y_train, y_test
        gc.collect()

    # create list of sliding windows that should be classified
    def create_search_windows(self, image):
        
        row, col, ch = image.shape
        print("Image shape ", row, col, ch)
        
        # slide huge windows across all area of interest (big vehicle appearence)
        windows_huge = slide_window(image, 
                                    x_start_stop=self.huge_x_start_stop, 
                                    y_start_stop=self.huge_y_start_stop, 
                                    xy_window=self.huge_xy_window, 
                                    xy_overlap=self.huge_xy_overlap)
                                    
        # slide large windows across upper part of road (large vehicle appearence)
        windows_large = slide_window(image, 
                                     x_start_stop=self.large_x_start_stop, 
                                     y_start_stop=self.large_y_start_stop, 
                                     xy_window=self.large_xy_window, 
                                     xy_overlap=self.large_xy_overlap)
        
        # slide small windows across horizont (small vehicle appearence)
        windows_small = slide_window(image, 
                                     x_start_stop=self.small_x_start_stop, 
                                     y_start_stop=self.small_y_start_stop, 
                                     xy_window=self.small_xy_window, 
                                     xy_overlap=self.small_xy_overlap)

        # Use window list to append window positions to
            
        self.windows.extend(windows_huge)
        self.windows.extend(windows_large)
        #self.windows.extend(windows_small)
        
        print("Number of total windows: ", len(self.windows))
        
        del windows_huge, windows_large, image
        gc.collect()
        
    # Pass an image and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img):
    
        # Create an empty list to receive positive detection windows
        true_windows = []
        
        # Iterate over all windows in the list
        for window in self.windows:
            
            # Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))   
            
            # Extract features for that window using single_img_features()
            features = single_img_features(test_img, 
                                           color_space=self.color_space, 
                                           spatial_size=self.spatial_size, 
                                           hist_bins=self.hist_bins, 
                                           orient=self.orient, 
                                           pix_per_cell=self.pix_per_cell, 
                                           cell_per_block=self.cell_per_block, 
                                           hog_channel=self.hog_channel, 
                                           spatial_feat=self.spatial_feat, 
                                           hist_feat=self.hist_feat, 
                                           hog_feat=self.hog_feat)
                                           
            # Scale extracted features to be fed to classifier
            test_features = self.X_scaler.transform(np.array(features).reshape(1, -1))
            
            # Predict using trained classifier
            prediction = self.classifier.predict(test_features)
            
            # If positive (prediction == 1) then save the window
            if prediction == 1:
                true_windows.append(window)
                
        #del img, test_img, features, test_features, prediction
        #gc.collect()
        
        # Return windows for positive detections
        return true_windows