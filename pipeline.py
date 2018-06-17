import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from lesson_functions import *
from scipy.ndimage.measurements import label
from time import time
from collections import deque



# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, cspace, hog_channel): # spatial_size, hist_bins):
    t_begin = time()
    
    if np.max(img) > 200:
        img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch) 
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    t_channels = time()
    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, tsqrt=True)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False, tsqrt=True)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False, tsqrt=True)
    
    t_hog = time()
    scores = list()
    features=list()
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel == "ALL":
                hog_features = np.dstack((hog_feat1, hog_feat2, hog_feat3))
            elif hog_channel == 0:
                hog_features = hog_feat1
            elif hog_channel == 1:
                hog_features = hog_feat2
            elif hog_channel == 2:
                hog_features = hog_feat3
            if np.any(np.isnan(hog_features)):
                continue

            features.append(hog_features.ravel())
            
    features = np.array(features)
    # Scale features and make a prediction
    test_features = X_scaler.transform(features)  
    test_predictions = svc.decision_function(test_features)
    t_predictions = time()
    bboxes_out = list()
    for i in range(len(test_predictions)):
        yb = np.mod(i, nysteps)
        xb = i // nysteps
        ypos = yb*cells_per_step
        xpos = xb*cells_per_step
        xleft = xpos*pix_per_cell
        ytop = ypos*pix_per_cell
        pred = test_predictions[i]
        if pred > 0.0:
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            bboxes_out.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
        t4 = time()
           
    t_end = time()
    
    return bboxes_out



# Define a function for extracting all the windows of a given ystart/ystop and scale
def create_windows(img, ystart, ystop, scale, orient, pix_per_cell, cell_per_block, cspace, hog_channel): # spatial_size, hist_bins):
    t_begin = time()
    if np.max(img) > 200:
        img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    if cspace != 'RGB':
        if cspace == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else: ctrans_tosearch = np.copy(img_tosearch) 
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    t_channels = time()
    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, tsqrt=True)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False, tsqrt=True)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False, tsqrt=True)
    
    t_hog = time()
    bboxes_out = list()
    test_features=0
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            if hog_channel == "ALL":
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            elif hog_channel == 0:
                hog_features = hog_feat1
            elif hog_channel == 1:
                hog_features = hog_feat2
            elif hog_channel == 2:
                hog_features = hog_feat3
                
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
  

            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            bboxes_out.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
           
    t_predictions = time()
    return bboxes_out, 
    
    
    
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        image = cv2.resize(image, (64, 64))
#         print("Min: ", np.min(image), " max:", np.max(image))
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True, tsqrt=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True, tsqrt=True)
        # Get color features
#         spatial_features = bin_spatial(image, size=(spatial, spatial))
#         hist_features = color_hist(image, nbins=histbin)
        
        # Append the new feature vector to the features list
        feature = np.hstack([hog_features])
        features.append(feature)
    # Return list of feature vectors
    return features


def add_heat(heatmap, bbox_list, inc_amounts=None):
    # Iterate through list of bboxes
    if inc_amounts is None:
        inc_amounts = np.ones(len(bbox_list))
    for i in range(len(bbox_list)):
        box = bbox_list[i]
        inc_amount = inc_amounts[i]
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += inc_amount

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_region_threshold(heatmap, threshold, labels):
    for car_number in range(1, labels[1]+1):
        # Find region labels for the image
        region = labels[0] == car_number
        # Find pixels with each car_number label value
        mval = np.max(heatmap[region])
        if mval < threshold:
            nonzero = (region).nonzero()
            # Zero out pixels below the threshold
            heatmap[nonzero] = 0
    # Return thresholded map
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


# The main pipeline function
history = deque(maxlen=10)
def pipeline(img, params, add_heatmap=False):  
    
    
    # ystop changes with the value of scale
    ystart = 400
    ystop = [528, 656, 656, 656]
    scale = [1.0, 1.5, 2.0, 2.5]

    # img_tosearch = img[ystart:656,:]
    
    # get attributes of our svc object
    svc = params["svc"]
    X_scaler = params["scaler"]
    orient = params["orient"]
    pix_per_cell = params["pix_per_cell"]
    cell_per_block = params["cell_per_block"]
    cspace = params["colorspace"]
    hog_channel = params["hog_channel"]
    
    
    bboxes_out = list()
    t1 = time()
    for j in range(len(scale)):
        bboxes = find_cars(img, ystart, ystop[j], scale[j], svc, X_scaler, orient, pix_per_cell, cell_per_block, cspace, hog_channel) #, spatial_size, hist_bins)
        bboxes_out += bboxes
    t2 = time()
    # Update array
    history.append(bboxes_out)
    
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    for bbox_list in history:
        # Add heat to each box in box list
        heat = add_heat(heat,bbox_list)
    t3 = time()
        
    # for i in range(len(bboxes_out)):
        # cv2.rectangle(draw_img, bbox[0], bbox[1], color=(0,0,255), thickness=6)

    
    # Create initial regions to filter out
    initial_labels = label(heat)

    # Apply threshold to help remove false positives
    heat_thresholded = apply_region_threshold(heat, 1, initial_labels) 

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat_thresholded, 0, 255) 
    labels = label(heatmap)
    
    # Find final boxes from heatmap using label function
    draw_img = np.copy(img)
    if np.max(draw_img) <= 1.0:
        draw_img = draw_img * 255
        draw_img = draw_img.astype(np.uint8)
    draw_img = draw_labeled_bboxes(draw_img, labels)
    
    
    # print("Time at pipeline stage 1", t2-t1)
    # print("Time at pipeline stage 2", t3-t2)
    # print("Time at pipeline stage 3", t4-t3)
    # print("Time at pipeline stage 4", t5-t4)
    if add_heatmap:
        return draw_img, heat
    else:
        return draw_img
