import os 
import numpy as np 

import cv2
from tqdm import tqdm

import math
import csv


activity='swing'
numpy_dir='/media/felicia/Data/object_detection/data/%s'%activity
numpy_name='image_bbgame_swing'
len_num_shards=6

outputfolder='/media/felicia/Data/object_detection/optical_flow/%s'%activity
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)


HEIGHT=720
WIDTH=1280


for i in range(1):
    numpy_path = os.path.join(numpy_dir,'%s-%s-of-%s.npy' % (numpy_name,str(i),str(len_num_shards)))

    print('Loading %s', numpy_path)

    numpy_dict= np.load(numpy_path,allow_pickle=True)
    numpy_dict=numpy_dict.item()

    images=numpy_dict['images']
    labels=numpy_dict['activity']
    steps=numpy_dict['steps']
    vd_names= numpy_dict['videos']

    nframes=len(images)

    ave_score=[] # average score of dense optical flow for each frame
    dense_flow=[] # dictionary of average dense optical flow

    for j in tqdm(range(nframes)):

        if steps[j]==0:
            prvs_rgb=images[j]
            prvs_gray = cv2.cvtColor(prvs_rgb,cv2.COLOR_BGR2GRAY)
            ave_score.append(0)
            dense_flow.append(np.zeros_like(prvs_rgb))
            hsv = np.zeros_like(prvs_rgb)
            hsv[...,1] = 255 
            continue

        next_rgb=images[j]
        next_gray=cv2.cvtColor(next_rgb,cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs_gray,next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0) 
        # pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags[, flow]

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2 # Hue
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX) # Value
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        ave_score.append(np.mean(rgb,axis=(0,1,2)))
        dense_flow.append(rgb)

        prvs_gray = next_gray

    output_filename = os.path.join(
        outputfolder,
        '%s-%s-of-%s.npy' % ('ori_flow_bbgame_swing_test',str(i),str(len_num_shards)))

    new_dict={
        'ave_score':np.array(ave_score), # np.array: B*1
        'dense_flow':dense_flow, # np.array:B*1
    }

    with open(output_filename,'wb') as file:
        np.save(file,new_dict)
