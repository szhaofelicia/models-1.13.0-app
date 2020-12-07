import os 
import numpy as np 

import cv2
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import math
import csv
import copy
import collections
import json

from utils import label_map_util
from utils import visualization_utils as vis_util

import yaml
from datetime import datetime
import copy

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
    windowsize=5

bbox_dir='/media/felicia/Data/object_detection/bbox/%s'%activity
bbox_name='bbox_bbgame_swing_person'
valid_name='valid_bbgame_swing'
patch_dir='/media/felicia/Data/object_detection/patch/%s'%activity

len_num_shards=6

WIDTH=1280
HEIGHT=720
patchH= 500  #350
patchW= 450  #450
ADDAVE=True
RMVAVE=True

max_boxes=20 # default: 10
min_occur= 5  # FP:3, FN: 5
min_iou=0.11 # default: 0.11

AREA_CONDITION=True
if AREA_CONDITION:
    max_area=0.25 # default: 1
    min_area=0.01
    area_name='area%s%s' % ("{:.0%}".format(max_area),"{:.0%}".format(min_area))
else:
    area_name=''
    
NORMALIZED_FLOW='QUANTILE' # 'NORMALIZED','ORIGINAL','QUANTILE'

min_flow=0.8
theta= math.atan(0.5)# 26.5 degree

bbox_fps=10
vd_fps=30
IMAGE_SIZE = (16, 8) # (w,h)

ox=WIDTH/2 
oy=HEIGHT 

now=datetime.now()
# now_str=now.strftime("%m-%d-%H-%M")
now_str='11-10-20-48' # min_iou=0.8

t=5
if NORMALIZED_FLOW == 'NORMALIZED':
    valid_path = os.path.join(bbox_dir,'%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(t),str(len_num_shards),flow_type,"{:.0%}".format(min_flow),area_name))
    outputfolder='/media/felicia/Data/object_detection/vis/patch/%s/%s'%(activity,now_str)
elif NORMALIZED_FLOW=='ORIGINAL':
    valid_path = os.path.join(bbox_dir,'%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(t),str(len_num_shards),flow_type,"{:02d}".format(min_flow),area_name))
    outputfolder='/media/felicia/Data/object_detection/vis/patch/%s/%s'%(activity,now_str)
else:
    valid_path = os.path.join(bbox_dir,'%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(t),str(len_num_shards),flow_type,"{:.2f}".format(min_flow)+'p',area_name))
    outputfolder='/media/felicia/Data/object_detection/vis/patch/%s/%s'%(activity,now_str)


if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

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
                box_to_color_map[box] = categorical_colors[i]
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


def crop_patches_thru_bbox(boxes):

    candidate_bbox=[]
    candidate_patch=[]
    l=len(boxes)

    nbbox=[len(bbox) for bbox in boxes]
    ref_fr=np.argmax(nbbox)

    ref_remove=[]
    keep_all=[]
    remove_all=[]
    print('ref_fr',ref_fr,l)
    for i in range(nbbox[ref_fr]):
        left_x, right_x, top_x, bottom_x=boxes[ref_fr][i][2]
        area_x=(left_x-right_x)*(top_x-bottom_x)
        pos_frame=[]
        neg_frame=[]
        buffer=[]
        buffer.append([left_x, right_x, top_x, bottom_x])
        for j in range(l):
            if j!= ref_fr:
                tmp_iou=[]
                for k in range(len(boxes[j])):
                    left_y, right_y, top_y, bottom_y=boxes[j][k][2]
                    area_y=(left_y-right_y)*(top_y-bottom_y)
                    inter_l,inter_r,inter_t,inter_b=np.max([left_x,left_y]),np.min([right_x,right_y]),np.max([top_x,top_y]),np.min([bottom_x,bottom_y])
                    if inter_l<=inter_r and inter_t <= inter_b:
                        intersection=(inter_r-inter_l)*(inter_b-inter_t)
                        iou=intersection/(area_x+area_y-intersection)
                    else:
                        iou=0
                    tmp_iou.append(iou)
                if tmp_iou:
                    track=np.argmax(tmp_iou)
                    # print(i,tmp_iou,track)
                    if tmp_iou[track]> min_iou:
                        pos_frame.append((j,track))
                        left_y, right_y, top_y, bottom_y=boxes[j][track][2]
                        buffer.append([left_y, right_y, top_y, bottom_y])
                    else:
                        neg_frame.append((j,None))
                else:
                    neg_frame.append((j,None))
        # print(i,pos_frame,neg_frame)
        if len(pos_frame)>min_occur:
            left_ave, right_ave, top_ave, bottom_ave=np.mean(buffer,axis=0)
            left_ave, right_ave, top_ave, bottom_ave=math.floor(left_ave),math.ceil(right_ave),math.floor(top_ave),math.ceil(bottom_ave)
            cx_ave,cy_ave=(left_ave+right_ave)//2, (top_ave+bottom_ave)//2
            cos_sim=np.inner([cx_ave-ox,cy_ave-oy],[WIDTH-ox,HEIGHT-oy])/(np.linalg.norm([cx_ave-ox,cy_ave-oy])*np.linalg.norm([WIDTH-ox,HEIGHT-oy]))
            keep_all+=pos_frame
            for idxf,_ in neg_frame:
                if ADDAVE:
                    print('add',(ref_fr,i),idxf)
                    boxes[idxf].append((cos_sim,(cx_ave,cy_ave),(left_ave, right_ave, top_ave, bottom_ave)))
        else:
            ref_remove.append((ref_fr,i))
            remove_all+=pos_frame
        # print(i,'ref_remove',ref_remove)


    remove_all+=ref_remove
    remove_all=list(set(remove_all))
    remove_all=sorted(remove_all,key=lambda x:(x[0],-x[1]))
    keep_all=set(keep_all)
    # print(i,'remove_all',remove_all)
    # print(i,'keep_all',keep_all)

    print('ref_remove',ref_remove if ref_remove else ref_fr)
    print('keep_all',keep_all)
    print('remove_all',remove_all)
    print('----------------------------------------------')
    if RMVAVE:
        for idxf,idxb in remove_all:
            if (idxf,idxb) not in keep_all:
                print('remove',idxf,idxb)
                boxes[idxf].pop(idxb) 


        
    for j in range(l):
        candidate_bbox.append(sorted(boxes[j],key=lambda x:(x[0])))
    
    for j in range(l):
        tmp_patch=[]
        lx,ly=candidate_bbox[j][0][1] if len(candidate_bbox[j])>0 else [WIDTH*0.25, HEIGHT*0.5] # leftmost bbox
        if patchW/2 <= lx <= WIDTH-patchW/2:
            l = lx-patchW/2
        elif lx <patchW/2:
            l = 0
        else:
            l= WIDTH-patchW
        if patchH/2 <= ly <= HEIGHT-patchH/2:
            t=ly-patchH/2
        elif ly <patchH/2:
            t=0
        else:
            t=HEIGHT-patchH
        r, b =l+patchW,t+patchH
        tmp_patch.append([l,r,t,b])

        right_boxes=[[right[1][0],right[1][1]] for right in candidate_bbox[j][1:]]
        if right_boxes:
            rx,ry=np.mean(right_boxes,axis=0) # right bbox
        else:
            rx,ry= WIDTH*0.75, HEIGHT*0.5
        rx,ry=np.round(rx),np.round(ry)
        if patchW/2 <= rx <= WIDTH-patchW/2:
            l = rx-patchW/2
        elif rx <patchW/2:
            l = 0
        else:
            l= WIDTH-patchW
        if patchH/2 <= ry <= HEIGHT-patchH/2:
            t=ry-patchH/2
        elif ry <patchH/2:
            t=0
        else:
            t=HEIGHT-patchH
        r, b =l+patchW,t+patchH
        tmp_patch.append([l,r,t,b])
        candidate_patch.append(tmp_patch)
    return candidate_bbox,candidate_patch





config={
    'HEIGHT':{
        'WIDTH': WIDTH,
        'HEIGHT': HEIGHT
    },
    'PATCH':{
        'patchH': patchH,
        'patchW': patchW,
        'ADDAVE': ADDAVE,
        'RMVAVE': RMVAVE
    },
    'BBOX':{
        'AREA_CONDITION': True,
        'MAX_AREA': max_area,
        'MIN_AREA': min_area,
        'MIN_OCCUR': min_occur,
        'MIN_IOU': min_iou,
        'THETA':theta
    },
    'OPTICAL_FLOW':{
        'FLOW_TYPE': NORMALIZED_FLOW,
        'MIN_FLOW': min_flow  
    }
}

# with open(outputfolder+'/config.yml', 'w') as file:
#     documents = yaml.dump(config, file)

# print("Configuration saved.")


# for i in range(1):

image_path = os.path.join(image_dir,'%s-%s-of-%s.npy' % (image_name,str(t),str(len_num_shards)))
print('Loading %s', image_path)
image_dict= np.load(image_path,allow_pickle=True)
image_dict=image_dict.item()

images=image_dict['images']
labels=image_dict['activity']
steps=image_dict['steps']
vd_names= image_dict['videos']

valid_bbox_dict= np.load(valid_path,allow_pickle=True)
valid_bbox_dict=valid_bbox_dict.item()

valid_bboxes=valid_bbox_dict['boxes']
valid_scores=valid_bbox_dict['scores']
valid_classes=valid_bbox_dict['classes']

nframes=len(images)

vd_dict={}
sorted_bbox=[]
for i in tqdm(range(nframes)):
    if i+1==nframes or steps[i+1]==0:
        vd_dict[vd_names[i]]=steps[i]+1
        # print(i,steps[i],steps[i],vd_dict[vd_names[i]])
    tmp_bbox=[]
    for j in range(len(valid_bboxes[i])):
        left, right, top, bottom=valid_bboxes[i][j]
        cx,cy=(left+right)//2, (top+bottom)//2
        cos_sim=np.inner([cx-ox,cy-oy],[WIDTH-ox,HEIGHT-oy])/(np.linalg.norm([cx-ox,cy-oy])*np.linalg.norm([WIDTH-ox,HEIGHT-oy]))
        if cos_sim<math.cos(theta):
            tmp_bbox.append((cos_sim,(cx,cy),(left, right, top, bottom)))
    sorted_bbox.append(sorted(tmp_bbox,key=lambda x:(x[0])))

i=0
crop_patch=[]
crop_bbox=[]
while i < nframes:
    st=i
    video=vd_names[i]
    length=vd_dict[video]
    ed=st+length

    print(i,length,vd_names[i],steps[st:ed])
    print('-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#')

    candi_bbox,patches=crop_patches_thru_bbox(copy.deepcopy(sorted_bbox[st:ed]))
    crop_patch+=patches
    crop_bbox+=candi_bbox

    print('--------------------------')
    print(length,len(patches),len(candi_bbox))


    i=ed


new_dict={
    'video':vd_names,
    'frame':steps,
    'sorted_bbox':sorted_bbox, # np.array: B*1
    'crop_bbox':crop_bbox, # np.array:B*1
    'crop_patch':crop_patch
}

with open(os.path.join(patch_dir,'%s-%s-of-%s_%s.npy' % ('patch_all',str(t),str(len_num_shards),now_str)),'wb') as file:
    np.save(file,new_dict)



## List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

for i in tqdm(range(nframes)):
    image_bbox0=copy.deepcopy(images[i])
    image_bbox1=copy.deepcopy(images[i])

    sorted_bbox_=[t[2] for t in sorted_bbox[i]]

    visualize_ordered_boxes_and_labels_on_image_array(
        image_bbox0,
        sorted_bbox_,
        None,
        None,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        skip_scores=True,
        skip_labels=True)
    
    visualize_ordered_boxes_and_labels_on_image_array(
        image_bbox1,
        crop_patch[i],
        None,
        None,
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        skip_scores=True,
        skip_labels=True)
    
    fig,ax=plt.subplots(1,2,figsize=IMAGE_SIZE)
    ax[0].imshow(image_bbox0)
    ax[1].imshow(image_bbox1)

    plt.suptitle('Video:%s Step:%s'%(vd_names[i],"{:.0%}".format(min_flow)))

    if NORMALIZED_FLOW == 'NORMALIZED':
        output_filename = os.path.join(
            outputfolder,
            '%s%s_%s.png' % (vd_names[i].decode("utf-8"),"{:04d}".format(steps[i]),"{:.0%}".format(min_flow)))
    elif NORMALIZED_FLOW == 'ORIGINAL':
        output_filename = os.path.join(
            outputfolder,
            '%s%s_%s.png' % (vd_names[i].decode("utf-8"),"{:04d}".format(steps[i]),"{:02d}".format(min_flow)))
    else:
        output_filename = os.path.join(
            outputfolder,
            '%s%s_%s.png' %  (vd_names[i].decode("utf-8"),"{:04d}".format(steps[i]),"{:.2f}".format(min_flow)+'p'))
    

    plt.savefig(output_filename,doi=100)
    plt.close()

    # print(i,vd_names[i].decode("utf-8"),"{:04d}".format(steps[i]))
