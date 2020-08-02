# 2D Object Detector for Waymo Open Dataset

## Project description
You can find here my implementation of 2D object detection system that can detect cars, pedestrians, etc.
The first model I based on YOLO.
The second one I intend to base on R-CNN network.

You can find commented code complemented by axuiliary functions like:
 - KPIs module measuring AP (Average Precision) across each class
 - Visualisation of comparision between prediction and ground truth for a specific image
 - Complilation of GIF or video of your predictions

![preview_static](assets/preview_jpg.jpg)

## Usage
You can find the code in notebook file `2D_object_recognition.ipynb`
The notebook is prepared to be run on Google Colab accessing data stored on your google drive.

Instructions to execute basic scenario:
1. Prepare folder structure on your google drive
Folder structure is preconfigured by default to be in this format:
- `your_folder/training`: contains waymo data in its original form of .tfrecord
- `your_folder/exctracted_gt` contains extracted images with corresponding ground truth in TXT file with the same name as the image. This folder is then used as input for training and predictions
- `your_folder/YOLO`: contains predicted object detections
2. Download [data](https://waymo.com/open/) and upload uncompressed folders into `training` folder.
So training folder will look like:
- `your_folder/training/training_0000/`  
- `your_folder/training/training_0001/` 
- ...
- `your_folder/training/training_n`   

3. Open notebook on your Google Colab and run code to extract ground truth from the dataset
This extracts from tfrecord video streams images with corresponding ground truth TXT files into folder `exctracted_gt`

Each object annotation is stored as separate line in this format:
`<object-class> <center_x> <center_y> <width> <height>` 
Example:
`1 809 719 89 73`
        
This ground truth is then used for KPI calculation of your prediction and training of the network.

5. Execute object detection
![preview_gif](assets/preview_gif.gif)

## Libraries used
Python 3
- numpy
- matplotlib
- TensorFlow
- OpenCV
