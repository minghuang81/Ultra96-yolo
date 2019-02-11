#!/usr/bin/env python
import numpy as np
import datetime
import os
import cv2
from lib import utils
from itf import convNetC

def detection_in_image(image_file):
    '''
    Argument
    image_file -- full or relative file path including file suffix, e.g. "./images/test.jpg"
    Return
    image_data -- RGB image of shape (h,w,3) e.g. (720,1024)
    '''

    # resize image to fit the model having a fixed size of h = w = 608 pixels
    image_data,resized_image = utils.preprocess_image_cv(image_file, model_image_size = (608, 608))
    image_shape = image_data.shape[0:2]
    
    # the outcome of object detections are boxes surrounding the found objects
    yolo_evals = convNetC.yolo(resized_image[0],anchors,len(class_names),image_shape)
        
    # yolo_evals = (score of each selected boxes, 4-coordinates of each box, best class for each selected box)
    out_scores, out_boxes, out_classes = yolo_evals[:,0:1].flatten(),yolo_evals[:,1:5],yolo_evals[:,5:6].astype('int').flatten()   

    # surround the found objects in image
    utils.draw_boxes(image_data, out_scores, out_boxes, out_classes, class_names)
     
    return image_data,out_boxes

# show prediction
# ---------------

dir_in = "./images"
dir_out = "./out"
class_names = utils.read_classes("model/coco_classes.txt")
anchors = utils.read_anchors("model/yolo_anchors.txt").astype('float32')

for filename in os.listdir(dir_in):
    if filename.endswith(".jpg"): 
        image_file=os.path.join(dir_in, filename)
        output_file=os.path.join(dir_out, filename)
        # perform detection with YOLO algo
        print(' ---------------- input image: {} in {}, output image: {} in {} -------------'.format(filename,dir_in,filename,dir_out))
        
        t1 = datetime.datetime.now()
        image,out_boxes = detection_in_image(image_file)
        t2 = datetime.datetime.now()
        # Print predictions info
        print('Have detected {} objects. consomed time: {} -> {}\n'.format(len(out_boxes), t1,t2))
        # save image with boxes over detected image
        utils.save_image(output_file, image)
    else:
        continue
