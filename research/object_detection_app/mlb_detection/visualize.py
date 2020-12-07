import os 
import numpy as np 

import cv2
from tqdm import tqdm

from utils import label_map_util
from utils import visualization_utils as vis_util

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import copy
import math
import collections
import csv


"""
Visualize average dense optical flow for each video


activity='swing'
numpy_dir='/media/felicia/Data/object_detection/optical_flow/%s'%activity
numpy_name='flow_bbgame_swing'
len_num_shards=6

outputfolder='/media/felicia/Data/object_detection/vis/ave_optical_flow/%s'%activity
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)


HEIGHT=720
WIDTH=1280

for i in range(len_num_shards):
    numpy_path = os.path.join(numpy_dir,'%s-%s-of-%s.npy' % (numpy_name,str(i),str(len_num_shards)))

    print('Loading %s', numpy_path)

    numpy_dict= np.load(numpy_path,allow_pickle=True)
    numpy_dict=numpy_dict.item()

    ave_score=numpy_dict['ave_score']
    flow_dict=numpy_dict['flow_dict']

    for vd in flow_dict.keys():
        flow=flow_dict[vd]
        vd=vd.decode("utf-8")

        output_filename = os.path.join(outputfolder,'%s.png' % vd)
        cv2.imwrite(output_filename,flow)

"""

"""
Visualize original dense optical flow for each video


activity='swing'
image_dir='/media/felicia/Data/object_detection/data/%s'%activity
image_name='image_bbgame_swing'

numpy_dir='/media/felicia/Data/object_detection/optical_flow/%s'%activity
numpy_name='ori_flow_bbgame_swing'
len_num_shards=6

outputfolder='/media/felicia/Data/object_detection/vis/ori_optical_flow/%s'%activity
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)


HEIGHT=720
WIDTH=1280

for i in range(1):
    image_path = os.path.join(image_dir,'%s-%s-of-%s.npy' % (image_name,str(i),str(len_num_shards)))
    print('Loading %s', image_path)
    image_dict= np.load(image_path,allow_pickle=True)
    image_dict=image_dict.item()

    labels=image_dict['activity']
    steps=image_dict['steps']
    vd_names= image_dict['videos']

    del image_dict

    numpy_path = os.path.join(numpy_dir,'%s-%s-of-%s.npy' % (numpy_name,str(i),str(len_num_shards)))

    print('Loading %s', numpy_path)

    numpy_dict= np.load(numpy_path,allow_pickle=True)
    numpy_dict=numpy_dict.item()

    ave_score=numpy_dict['ave_score']
    dense_flow=numpy_dict['dense_flow']

    nframes=len(ave_score)

    for j in tqdm(range(nframes)):
        vd=vd_names[j]
        flow=dense_flow[j]
        vd=vd.decode("utf-8")

        output_filename = os.path.join(outputfolder,'%s%s.png' % (vd,"{:04d}".format(steps[j])))
        cv2.imwrite(output_filename,flow)

"""

"""

Visualize Bbox
"""

def visualize_ordered_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=0,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False):

    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    categorical_colors= ['darkorchid','mediumorchid','violet','plum','mediumpurple',
        'royalblue','deepskyblue','darkturquoise','paleturquoise','mediumspringgreen',
        'lightseagreen','seagreen','olivedrab','darkkhaki','gold',
        'moccasin','orange','darkorange','coral','orangered'] #20

    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    if not max_boxes_to_draw:
        max_boxes_to_draw = len(boxes)
    for i in range(min(max_boxes_to_draw, len(boxes))):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i])
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
            if not skip_labels:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = str(class_name)
            if not skip_scores:
                if not display_str:
                    display_str = '{}%'.format(int(100*scores[i]))
                else:
                    display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
            box_to_display_str_map[box].append(display_str)
            if agnostic_mode:
                box_to_color_map[box] = 'DarkOrange'
            else:
                box_to_color_map[box] = categorical_colors[i]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        left, right, top, bottom=box
        ymin, xmin, ymax, xmax = top/HEIGHT,left/WIDTH, bottom/HEIGHT,right/WIDTH
        vis_util.draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)

    return image


activity='swing'
HEIGHT=720
WIDTH=1280
IMAGE_SIZE = (24, 16) 
min_flow=10

NORMALIZED_FLOW='QUANTILE' # # 'NORMALIZED','ORIGINAL','QUANTILE'
if NORMALIZED_FLOW == 'NORMALIZED':
    flow_thresh= np.arange(0.07,0.13,0.01) #[0.01,0.03,0.05,0.07,0.1,0.13,0.15,0.17,0.2] #[0.01,0.05,0.1,0.15,0.2]
elif NORMALIZED_FLOW=='ORIGINAL':
    flow_thresh=[1,3,5,7,10]
else:
    flow_thresh= np.arange(0.78,0.86,0.01)



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

max_area=1
AREA_CONDITION=True
if AREA_CONDITION:
    max_area=0.25 # default: 1
    # area_name='area%s' % ("{:.0%}".format(max_area))
    min_area=0.01
    area_name='area%s%s' % ("{:.0%}".format(max_area),"{:.0%}".format(min_area))
else:
    area_name=''

bbox_dir='/media/felicia/Data/object_detection/bbox/%s'%activity
bbox_name='bbox_bbgame_swing_person'
valid_name='valid_bbgame_swing'

len_num_shards=6

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)



for i in range(1):
    image_path = os.path.join(image_dir,'%s-%s-of-%s.npy' % (image_name,str(i),str(len_num_shards)))
    print('Loading %s', image_path)
    image_dict= np.load(image_path,allow_pickle=True)
    image_dict=image_dict.item()

    images=image_dict['images']
    steps=image_dict['steps']
    vd_names= image_dict['videos']

    bbox_path = os.path.join(bbox_dir,'%s-%s-of-%s.npy' % (bbox_name,str(i),str(len_num_shards)))
    bbox_dict= np.load(bbox_path,allow_pickle=True)
    bbox_dict=bbox_dict.item()
    
    all_bboxes=bbox_dict['boxes']
    all_scores=bbox_dict['scores']
    all_classes=bbox_dict['classes']

    for f in tqdm(flow_thresh):

        if NORMALIZED_FLOW == 'NORMALIZED':
            valid_path = os.path.join(bbox_dir,'%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(i),str(len_num_shards),flow_type,"{:.0%}".format(f),area_name))
            outputfolder='/media/felicia/Data/object_detection/vis/bbox/%s/%s%s%s'%(activity,flow_type,"{:.0%}".format(f),area_name)
        elif NORMALIZED_FLOW=='ORIGINAL':
            valid_path = os.path.join(bbox_dir,'%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(i),str(len_num_shards),flow_type,"{:02d}".format(f),area_name))
            outputfolder='/media/felicia/Data/object_detection/vis/bbox/%s/%s%s%s'%(activity,flow_type,"{:02d}".format(f),area_name)
        else:
            valid_path = os.path.join(bbox_dir,'%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(i),str(len_num_shards),flow_type,"{:.2f}".format(f)+'p',area_name))
            outputfolder='/media/felicia/Data/object_detection/vis/bbox/%s/%s%s%s'%(activity,flow_type,"{:.2f}".format(f)+'p',area_name)
        

        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)

        valid_bbox_dict= np.load(valid_path,allow_pickle=True)
        valid_bbox_dict=valid_bbox_dict.item()
        
        valid_bboxes=valid_bbox_dict['boxes']
        valid_scores=valid_bbox_dict['scores']
        valid_classes=valid_bbox_dict['classes']
        
        flow_path = os.path.join(flow_dir,'%s-%s-of-%s.npy' % (flow_name,str(i),str(len_num_shards)))
        flow_dict= np.load(flow_path,allow_pickle=True)
        flow_dict=flow_dict.item()

        ave_scores=flow_dict['ave_score']
        if FILTERING_TYPE == 'AVERAGE':
            flow=flow_dict['flow_dict']
        elif FILTERING_TYPE == 'ORIGINAL':
            flow=flow_dict['dense_flow']

        nframes=len(images)

        for j in tqdm(range(nframes)):
            image_bbox0=copy.deepcopy(images[j])
            image_bbox1=copy.deepcopy(images[j])

            if FILTERING_TYPE=='AVERAGE':
                flow_rgb=np.float32(flow[vd_names[j]])
            elif FILTERING_TYPE=='ORIGINAL':
                flow_rgb=np.float32(flow[j])
            flow_gray=cv2.cvtColor(flow_rgb,cv2.COLOR_BGR2GRAY)

            visualize_ordered_boxes_and_labels_on_image_array(
                image_bbox0,
                all_bboxes[j],
                all_classes[j],
                all_scores[j],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20, # next time: change to 20
                min_score_thresh=0)

            visualize_ordered_boxes_and_labels_on_image_array(
                image_bbox1,
                valid_bboxes[j],
                valid_classes[j],
                valid_scores[j],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=0)
            
            if NORMALIZED_FLOW:           
                fmax,fmin=flow_gray.max(),flow_gray.min()
                flow_gray=(flow_gray - fmin)/(fmax- fmin)*255

            fig,ax=plt.subplots(2,2,figsize=IMAGE_SIZE)
            ax[0][0].imshow(images[j])
            ax[0][1].imshow(flow_gray)
            ax[1][0].imshow(image_bbox0)
            ax[1][1].imshow(image_bbox1)

            plt.suptitle('Video:%s Step:%s'%(vd_names[j],"{:.0%}".format(f)))

            if NORMALIZED_FLOW == 'NORMALIZED':
                output_filename = os.path.join(
                    outputfolder,
                    '%s%s_%s.png' % (vd_names[j].decode("utf-8"),"{:04d}".format(steps[j]),"{:.0%}".format(f)))
            elif NORMALIZED_FLOW == 'ORIGINAL':
                output_filename = os.path.join(
                    outputfolder,
                    '%s%s_%s.png' % (vd_names[j].decode("utf-8"),"{:04d}".format(steps[j]),"{:02d}".format(f)))
            else:
                output_filename = os.path.join(
                    outputfolder,
                    '%s%s_%s.png' %  (vd_names[j].decode("utf-8"),"{:04d}".format(steps[j]),"{:.2f}".format(f)+'p'))

            plt.savefig(output_filename,doi=100)
            plt.close()



