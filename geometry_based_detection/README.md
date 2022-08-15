# Geometry based detection

A python package to extract truss features from a given image and determined an optimal grasping pose.

## 1 Installation
Install the dependencies with the requirements.txt file:
```
cd ~/flexcraft_ws/src/vine_tomato_grasping/geometry_based_detection
pip install -r requirements.txt
```

### 1.3 Verify Instalation
Run the tests:
```
cd ~/flexcraft_ws/src/vine_tomato_grasping/geometry_based_detection/src
python -m unittest discover flex_vision/test
```

## 2 General introduction
Feature extraction:

<p float="center">
  <img src="doc/data/003.png" alt="input image" width="400"/>
  <img src="doc/results/003_original.png" alt="input image" width="400"/>
</p>

Moreover, if you provide a ground truth, an error may be determined:

<p float="center">
  <img src="doc/results/003_pend_error.png" alt="input image" width="400"/>
  <img src="doc/results/003_tom_error.png" alt="input image" width="400"/>
</p>

## 3 Code structure
Currently, this repository is fairly unstructured:

1. The main file for feature extraction and grasp pose determination is [detect_truss/ProcessImage.py](src/flex_vision/detect_truss/ProcessImage.py)
2. The main file for comparing these results with the ground truth is [detect_truss/analyze_results.py](src/flex_vision/detect_truss/analyze_results.py)

There are some additional files, first of these files profide functions used in ProcessImage.py
- [bbox_detection](src/flex_vision/bbox_detection/bbox_detection.py): detects tomato trusses from an input image, returns bbox coordinates
- [compute_grasp](src/flex_vision/detect_truss/compute_grasp.py): contains parameters used for computing the grasp location
- [detect_peduncle_2](src/flex_vision/detect_truss/detect_peduncle_2.py): contains parameters and functions for peducnle detection, NOTE: detect_peduncle.py is not used
- [detect_tomato](src/flex_vision/detect_truss/detect_tomato.py): contains parameters and functions for tomato detection
- [filter_segments](src/flex_vision/detect_truss/filter_segments.py): contains parameters and functions for filering segments
- [segment_image](src/flex_vision/detect_truss/segment_image.py): contains parameters and functions for image segmentation

There are two helper modules:
- [geometry](src/flex_vision/utils/geometry.py): this module contains the Point2D and Transform class, these are used to track 2d points in images with respect to different reference frames. This is usefull for ProcessImage.py since the actual image processing is done on a rotated and translated reference frame.
- [imgpy](src/flex_vision/utils/imgpy.py): this module contains the Image class, this is used for easy rotating and cropping of images, also used in ProcessImage.py

There are some tests:
- [test](src/flex_vision/detect_truss/test.py) contains proper tests for the geometry module. It certainly does not cover all possible scenaro but its a good start.
- [test_rotate_image](src/tests/test_rotate_image.py) contains a visual check for the geometry module.
- [tesy](src/tests) contains some files for running certain parts of the computer vision pipeline.

## conventions

All image coordinates are stated as (x,y) as this is also the default for open_cv and matplotlib. This means that for numpy these coordinates need to be swapped to get (row, column). 

## Data

### Process image
ProcessImage requires the following data per image:

1. An rgb image called [id].png
2. A [id]_info.json file containing the field px_per_mm which describes the number of pixels per millimeter present at the truss level

ProcessImage will generate a prediction file .json file called [id].json, in the results/dataset/json folder. This file contains information about the predicted tomatoes, peduncle and grasp location.

### analyze_results
Analyze results requires the folowing data:

1. A ground truth [id].json file containing the actual features
2. A prediction [id].json gile containing the predicted predicted features