import os 
import numpy as np 

import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import math
import csv

activity='swing'
image_dir='/media/felicia/Data/object_detection/data/%s'%activity
image_name='image_bbgame_swing'

flow_dir='/media/felicia/Data/object_detection/optical_flow/%s'%activity
FILTERING_TYPE= 'AVERAGE' # 'AVERAGE','ORIGINAL','WINDOW'


if FILTERING_TYPE=='AVERAGE':
    flow_name='ave_flow_bbgame_swing'
    flow_type='aveflow'
elif FILTERING_TYPE=='ORIGINAL':
    flow_name='wind_flow_bbgame_swing'
    flow_type='oriflow'
else:
    flow_name='wnd_flow_bbgame_swing'
    flow_type='wndflow'
    buffer=5

bbox_dir='/media/felicia/Data/object_detection/bbox/%s'%activity
bbox_name='bbox_bbgame_swing_person'
valid_name='valid_bbgame_swing'

len_num_shards=6

WIDTH=1280
HEIGHT=720
BATCH=50

max_boxes=20 # default: 10
min_score=.3
min_area=0 # default:0
max_area=1

AREA_CONDITION=True
if AREA_CONDITION:
    max_area=0.25 # default: 1
    # area_name='area%s' % ("{:.0%}".format(max_area))
    min_area=0.01
    area_name='area%s%s' % ("{:.0%}".format(max_area),"{:.0%}".format(min_area))
else:
    area_name=''
    
NORMALIZED_FLOW='QUANTILE' # # 'NORMALIZED','ORIGINAL','QUANTILE'
if NORMALIZED_FLOW == 'NORMALIZED':
    flow_thresh= np.arange(0.05,0.16,0.01) #[0.01,0.03,0.05,0.07,0.1,0.13,0.15,0.17,0.2]
elif NORMALIZED_FLOW=='ORIGINAL':
    flow_thresh=[1,3,5,7,10]
else:
    flow_thresh= np.arange(0.71,0.9,0.01)


def filtered_boxes_on_minibatch_images(videos,boxes,scores,classes,flow,min_score_thresh,min_area,max_area,min_flow,normalized_flow=False,average_flow=True):
    """
    Args:
    videos: List of strings
    boxes: List of n*4 np.array
    classes: List of n*1 np.array
    scores: List of n*1 np.array
    flow: H*W
    """
    valid_output={}
    valid_boxes=[]
    valid_scores=[]
    valid_classes=[]
    b=len(boxes)

    for i in range(b):
        vd=videos[i]
        if average_flow:
            flow_rgb=np.float32(flow[vd])
        else:
            flow_rgb=np.float32(flow[i])

        flow_gray=cv2.cvtColor(flow_rgb,cv2.COLOR_BGR2GRAY)

        if normalized_flow=='NORMALIZED' or 'QUANTILE' :
            fmax,fmin=flow_gray.max(),flow_gray.min()
            flow_gray=(flow_gray - fmin)/(fmax- fmin)

        if normalized_flow== 'QUANTILE':
            minflow=np.quantile(flow_gray,min_flow,axis=(0,1))
        else:
            minflow=min_flow

        nbox=len(boxes[i])
        tmp_boxes=[]
        tmp_scores=[]
        tmp_classes=[]

        for j in range(nbox):
            left, right, top, bottom=boxes[i][j]
            left, right, top, bottom=math.floor(left),math.ceil(right),math.floor(top),math.ceil(bottom)
            area=(left-right)*(top-bottom)

            bbox_flow=np.sum(flow_gray[top:bottom,left:right])/area

            if min_area< area/(WIDTH*HEIGHT) < max_area and bbox_flow>minflow and scores[i][j]>min_score_thresh:
                tmp_boxes.append(boxes[i][j])
                tmp_scores.append(scores[i][j])
                tmp_classes.append(classes[i][j])
        valid_boxes.append(tmp_boxes)
        valid_scores.append(tmp_scores)
        valid_classes.append(tmp_classes)

    valid_output['boxes']=valid_boxes
    valid_output['scores']=valid_scores
    valid_output['classes']=valid_classes

    return valid_output



print('Filter bbox with NORMALIZED_FLOW', NORMALIZED_FLOW)
print('Filter bbox with filtering type:', FILTERING_TYPE)
print('Filter bbox with Area condition', area_name)
print('Candiate boundaries', flow_thresh)


for i in range(1,len_num_shards):

    image_path = os.path.join(image_dir,'%s-%s-of-%s.npy' % (image_name,str(i),str(len_num_shards)))
    print('Loading %s', image_path)
    image_dict= np.load(image_path,allow_pickle=True)
    image_dict=image_dict.item()

    images=image_dict['images']
    labels=image_dict['activity']
    steps=image_dict['steps']
    vd_names= image_dict['videos']

    bbox_path = os.path.join(bbox_dir,'%s-%s-of-%s.npy' % (bbox_name,str(i),str(len_num_shards)))
    bbox_dict= np.load(bbox_path,allow_pickle=True)
    bbox_dict=bbox_dict.item()

    bboxes=bbox_dict['boxes']
    scores=bbox_dict['scores']
    classes=bbox_dict['classes']
    
    flow_path = os.path.join(flow_dir,'%s-%s-of-%s.npy' % (flow_name,str(i),str(len_num_shards)))
    flow_dict= np.load(flow_path,allow_pickle=True)
    flow_dict=flow_dict.item()

    ave_scores=flow_dict['ave_score']
    if FILTERING_TYPE == 'AVERAGE':
        flow=flow_dict['flow_dict']
    elif FILTERING_TYPE == 'ORIGINAL':
        flow=flow_dict['dense_flow']

    nframes=len(images)
    nbatch=nframes//BATCH+(1 if nframes%BATCH>0 else 0)

    # for f in tqdm(flow_thresh):
    for f in tqdm([0.80]):
        valid_boxes=[]
        valid_scores=[]
        valid_classes=[]

        for j in tqdm(range(nbatch)):
            idx_st=j*BATCH 
            idx_ed=(j+1)*BATCH if (j+1)*BATCH < nframes else nframes

            image_batch=images[idx_st:idx_ed] # B*H*W*C
            vd_batch=vd_names[idx_st:idx_ed]
            bbox_batch=bboxes[idx_st:idx_ed]
            score_batch=scores[idx_st:idx_ed]
            class_batch=classes[idx_st:idx_ed]

            if FILTERING_TYPE == 'AVERAGE':
                dense_flow=flow
            elif FILTERING_TYPE == 'ORIGINAL':
                dense_flow=flow[idx_st:idx_ed]

            valid_output=filtered_boxes_on_minibatch_images(
                vd_batch,
                bbox_batch,
                score_batch,
                class_batch,
                dense_flow,
                min_score,
                min_area,
                max_area,
                f,
                normalized_flow=NORMALIZED_FLOW,
                average_flow=FILTERING_TYPE
            )
            
            valid_boxes+=valid_output['boxes']
            valid_scores+=valid_output['scores']
            valid_classes+=valid_output['classes']
        

        if NORMALIZED_FLOW == 'NORMALIZED':
            output_filename = os.path.join(
                bbox_dir,
                '%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(i),str(len_num_shards),flow_type,"{:.0%}".format(f),area_name))
        elif NORMALIZED_FLOW=='ORIGINAL':
            output_filename = os.path.join(
                bbox_dir,
                '%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(i),str(len_num_shards),flow_type,"{:02d}".format(f),area_name))
        else:
            output_filename = os.path.join(
                bbox_dir,
                '%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(i),str(len_num_shards),flow_type,"{:.2f}".format(f)+'p',area_name))

        new_dict={
            'boxes':valid_boxes, # np.array: B*1
            'scores':valid_scores, # np.array:B*1
            'classes':valid_classes
        }

        with open(output_filename,'wb') as file:
            np.save(file,new_dict)



