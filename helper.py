#!/usr/bin/env python
#  -*- coding: utf-8 -*-

import cv2
import os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import xml.etree.ElementTree as ET
from generate_structure_tensor_lab import compute_structure_tensor
from data_extract import reconstruct_chart
from Retrieve_Text import remove_text, remove_legend
from Graph_Obj_Seg import segment
from Chart_Classification.model_loader import classifyImage
from summary import summaryGen

# Add border to chart objects
def add_border(seg_img):
    g_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    cedge_leftcorners = cv2.Canny(g_img,100,200)
    cedge_rightcorners = cv2.flip(cv2.Canny(cv2.flip(g_img, 1),100,200),1)
    cedge = cv2.bitwise_or(cedge_leftcorners, cedge_rightcorners, mask = None)
    contours, hierarchy = cv2.findContours(cedge,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    im = cv2.drawContours(seg_img, contours, -1, (0,0,0), 2)
    # print("Added Borders.....!")

    return im

# Read image and compute tensor field
def image_read(filename,xmlname):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    im = Image.open(filename)
    img = cv2.imread(filename)
    #From chart classification module
    chart_type=classifyImage(filename)
    if chart_type =='other':
        return 'other'
    h,w,_= np.shape(img)
    seg_img = 255 - np.zeros((h,w,3), dtype=np.uint8)
    # extract canvas
    root = ET.parse(xmlname).getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name=='canvas':
            box = obj.find('bndbox')
            (x0,y0)=(int(box.find('xmin').text),int(box.find('ymin').text))
            (x1,y1)=(int(box.find('xmax').text),int(box.find('ymax').text))
            seg_img[y0:y1,x0:x1,:] = segment(img[y0:y1,x0:x1,:], chart_type)
    remove_legend(seg_img,root)
    image = add_border(seg_img)
    print("Preprocessed "+filename)
    image_clr = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    image_smth = cv2.GaussianBlur(image_clr, (3, 3), 1)
    image2 = image_smth[::-1, :, :]
    data = write_image_to_csv(image2, filename)
    # compute structure tensor
    compute_structure_tensor(data, image2, filename)
    reconstruct_chart(filename,chart_type)
    summaryGen(filename)
    return chart_type

# Writing xy-coordinates and RGB values of image in csv
def write_image_to_csv(image, filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    with open(path+"Image_RGB_"+image_name+".csv", "w+") as my_csv:
        writer = csv.DictWriter(my_csv, fieldnames=["X", "Y", "Red", "Green", "Blue"])
        writer.writeheader()
        y = 0
        for row in image:
            x = 0
            for col in row:
                my_csv.write("%d, %d, " % (x, y))
                for val in col:
                    my_csv.write("%d," % val)
                my_csv.write("\n")
                x += 1
            y += 1
    data = pd.read_csv(path+"Image_RGB_"+image_name+".csv", sep=",", index_col=False)
    print("image file generated.")
    return data
