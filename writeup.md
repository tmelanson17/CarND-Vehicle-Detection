## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.png
[image4]: ./examples/sliding_window.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!


The project implementation is done by running the second cell video_pipeline.ipynb.

For the final result video, I used [this implementation]() of SSD (pre-trained) to obtain the result video. The main pipeline used was ```pipeline_ssd.py```.

Other things I tried was using the sliding-window implementation discussed in the class. A non-deep pipeline is located in the ```pipeline``` function of `pipeline.py`. Because the rubric wanted me to discuss the shallow learning + sliding window implementation, I did so in this writeup.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step (the hog function itself) is contained in lines 13 through 32 of ```pipeline.py```. The full feature extraction can be found in the first code cell in the ```classifier.ipynb``` file.  

Because reading the full datasets took so much time, I started by reading in the `vehicles_smallset` and `non-vehicles_smallset` images. However, when implementing the final video classifier, I used the entire `vehicles` and `non-vehicles` dataset, as well as a few non-vehicles used for hard negative mining Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

When looking at each of the colorspaces, I realized that intensity was the best distinguisher for vehicles. Both the Y values for the YCrCb color space and the L values in HLS did this. However, using the YCrCb colorspace achieved a slightly higher overall accuracy (96 to 99 percent as opposed to 93 to 94 percent).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using a GridSearch algorithm across several values of kernel ('linear', 'rbf') and C (0.01, 0.1, 1., 10.). I used the hog features that achieved the highest test accuracy: 8 pixels per block, 9 orientations, 2 blocks per cell, YCrCb colorspace, and using all channels.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For the test images, it looked like 96x96 window was big enough for the bigger cars, but too small to measure the smaller cars with any level of granularity. Because of this, I decided to add a slightly smaller scale (1.0), which covers the more distant cars but isn't small enough to take over 1000 windows per image.

Here is an example of all the boxes mapped onto the image.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features, which provided a nice result.  Here are some example images (done using the third and fourth code blocks of `writeup.ipynb`:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./results.mp4)

The code for the video implementation can be found in `video_pipeline.ipynb`.

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap for the blobs. When doing a simple threshold, I found that many of the output bounding boxes were broken, even with a 75% overlap. Because of this, I thresholded the **regions** themselves. In other words, I zeroed all continuous regions (found with `scipy.ndimage.measurements.label`) with no peak value, and kept the entirety of regions that had peaks above the maximum value. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are five frames (from test_images) and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


For the SSD framework, I would implement time-based filtering to improve accuracy, albeit a different form involving neighboring detections.

The first issue I faced was the lack of positive data. I improved upon this by improving the classifier until positive results were distinguishable. Second was the relatively high ranking of guard rails and shadows. I tried to solve this in two ways: first, by adding transform_sqrt to remove effects by shadows, and by hard negative mining. The transform_sqrt was effective agains shadows, but hard negative mining was not as helpful due to the relatively small number of samples I could create. 

Getting the detector to find the images throughout the video also proved difficult. I found that, in noisier images, the detector had a difficult time detecting the vehicles with a high confidence level. To solve this, I used the time filter not only to filter out data noise, but to "hold" true vehicle values for longer.

I also believe improving upon machine learning with either spatial / histogram filters, as well as using deep learning methods such as SSD could help.
