#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

@author: Hankui Peng

"""


## import necessary modules 
# system 
from skimage.segmentation import mark_boundaries
from skimage import io
import skimage
from tqdm import tqdm
from glob import glob 
import numpy as np
import argparse 
import random 
import torch
import time 
import sys 
import cv2
import os 

# local
sys.path.insert(0, "../pybuild")
sys.path.insert(0, "pybuild")
from utils.analysis_util import *
from utils.hed_edges import *
from model.network import *
import hers_superpixel

random.seed(100)


## main function 
def superpixel(nC, pretrained, input_dir, output_dir, output_suff, edge, device):
    
    data_type = np.float32
    
    # read all the image files in the folder
    tst_lst = glob(input_dir + '*.jpg')
    tst_lst.sort()

    # load the model 
    network_data = torch.load(pretrained, map_location=device)
    model = DAL(nr_channel=8, conv1_size=7)
    model.load_state_dict(network_data['state_dict'])
    model.to(device)
    model.eval()
    
    # for each image:
    for n in tqdm(range(len(tst_lst))):
            
        ## input image 
        img_file = tst_lst[n]
        imgId = os.path.basename(img_file)[:-4]
        image = cv2.imread(img_file)
        input_img = image.astype(data_type)
        h, w, ch = image.shape
        
        
        ## input affinities
        with torch.no_grad():
            affinities = ProduceAffMap(img_file, model)
            input_affinities = affinities
         
        
        ## HED edge information 
        if edge:
            Input = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(img_file))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)))*(1.0 / 255.0)
            edge_prob = estimate(Input)
            input_edge = np.array(edge_prob.cpu().squeeze(0), dtype=data_type)
        else:
            input_edge = np.ones((h, w), dtype=data_type) # Provide no external edge information by default 

        
        ## build the hierarchical segmentation tree 
        start = time.time()
        bosupix = hers_superpixel.BoruvkaSuperpixel()
        bosupix.build_2d(input_img, input_affinities, input_edge)
        end = time.time()
        
        
        ## segmentation with user-defined number of superpixels
        sp_label = bosupix.label(nC)
        output_img = np.max(image) * mark_boundaries(image, sp_label.astype(int), color = (0,0,255)) # candidate color choice: (220,20,60)
        
        # output the label map as a csv file 
        save_csv_path = output_dir + str(nC) + '/csv/'
        os.makedirs(save_csv_path, exist_ok=True)
        label_map_path = save_csv_path + imgId + output_suff + '.csv'
        np.savetxt(label_map_path, sp_label.astype(int), fmt='%i', delimiter=",") 
        
        # output the visualisation    
        save_png_path = output_dir + str(nC) + '/png/'
        os.makedirs(save_png_path, exist_ok=True)
        spixl_save_name = save_png_path + imgId + output_suff + '.png'
        cv2.imwrite(spixl_save_name, output_img)
        
        # save the run times 
        elapsed_time = (end - start)*1000



def progress_bar(progress, total):
    percent = 100 * (progress / float(total))
    bar = '█' * int(percent) + '-' * (100 - int(percent))
    print(f"\r|{bar}| {percent:.2f}%", end="\r")
