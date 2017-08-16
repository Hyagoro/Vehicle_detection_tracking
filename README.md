**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./readme/car_notcar.png
[image4]: ./readme/hog_feat.png
[image5]: ./readme/boxes.png
[image6]: ./readme/heat_map.png
[image7]: ./readme/heat_map2.png
[image8]: ./readme/heat_map3.png
[image9]: ./readme/heat_map4.png
[image10]: ./readme/heat_map5.png
[image11]: ./readme/heat_map6.png
[video1]: ./project_video.mp4

---

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the code cell 10 of the Jupyter notebook. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image4]

I tried various combinations of parameters (`orientations=11` and `pixels_per_cell=(16, 16)` but my results was interesting.

#### Classifier training using HOG features, binned color features and histogram.

I trained a Random Forest classifier insteed of LinearSVM because I think Random Forest is a better generalizer and the learning time is almost the same. I also tried SVM but the learning time was far too long. With the Random forest (max_depth=18) I got 0.9842 accuracy.

### Sliding Window Search

I decided to search with different window positions and scales following this array:

| Scales | Coordinates |
| ------ | ----------- |
| 2      | [380, 528]  |
| 1.5    | [380, 496]  |
| 1.5    | [415, 542]  |
| 1      | [410, 470]  |
| 1      | [400, 480]  |
| 0.7    | [400, 480]  |

I tried to use PCA to reduce feature vector length (8460) to 4000, and 6000 but the results wasn't good enough due to loosing a lot of calculation time during prediction.
Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][image5]
---

### Video Implementation

Here's a [link to my video result](./project_video_processed_met2.mp4)


I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

---

### Discussion

A big part of this project was to find where I had to put my sliding windows in order to be at a good scale regarding the road. I think it's still too long to process, I need to find better coordinates to improve my pipeline, mostly because my boxes are to small comparing to the vehicle size. I spend alse a lot of time finding the best color transformation to use to detect vehicles (HLS, YUV, YCrCb, RGB, ...).

I also implement two classes in note 2 of the Jupyter notebook. This 2 classes helps to remove outliers by giving a increasing confidence to "hot points" from the heatmap during video processing.

To compare my project to the state of the art in pattern detection (very frustrating), I found YOLO a deep neural network based on COCO dataset, which have incredible perfomances in this field.

